import os
import queue
import time
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from openai import OpenAI

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.functionals import concatenate_input_and_output, GenerateRequestType
from roll.utils.logging import get_logger

logger = get_logger()


class ApiInferStrategy(InferenceStrategy):
    strategy_name = "api_infer"

    def __init__(self, worker: "Worker"):
        super().__init__(worker)
        self.uses_cuda = False
        self.client = None
        self.model_name = None
        self.strip_special_tokens = True
        self.system_message = None
        self.max_retries = 2
        self.timeout = 60
        self.api_key_env = "OPENAI_API_KEY"
        self.command_queue: queue.Queue = queue.Queue()
        self.running = False
        self.aborted_request_ids = set()

    def initialize(self, model_provider):
        strategy_config = self.worker_config.strategy_args.strategy_config or {}
        base_url = strategy_config.get("base_url")
        if not base_url:
            raise ValueError("api_infer requires strategy_config.base_url")

        self.api_key_env = strategy_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"api_infer requires API key in env var {self.api_key_env} (set it before running)."
            )

        self.model_name = strategy_config.get("model") or strategy_config.get("model_name") or "gpt-4o"
        self.strip_special_tokens = bool(strategy_config.get("strip_special_tokens", True))
        self.system_message = strategy_config.get("system_message")
        self.max_retries = int(strategy_config.get("max_retries", 2))
        self.timeout = int(strategy_config.get("timeout", 60))

        tokenizer_path = strategy_config.get("tokenizer_path")
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                split_special_tokens=False,
                trust_remote_code=True,
                padding_side="left",
            )
        else:
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"api_infer initialized with base_url={base_url}, model={self.model_name}")

    def forward_step(self, *args, **kwargs):
        raise NotImplementedError("api_infer does not support forward_step/log_probs.")

    def generate(self, batch: DataProto, generation_config):
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        prompts = self._decode_prompts(input_ids, attention_mask)

        num_return_sequences = int(generation_config.get("num_return_sequences", 1))
        max_new_tokens = generation_config.get("max_new_tokens") or generation_config.get("max_length")
        do_sample = generation_config.get("do_sample", True)
        temperature = generation_config.get("temperature", 1.0) if do_sample else 0.0
        top_p = generation_config.get("top_p")

        output_texts: List[str] = []
        for prompt in prompts:
            output_texts.extend(
                self._complete_prompt(
                    prompt=prompt,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

        response_tensors: List[torch.Tensor] = []
        for text in output_texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if max_new_tokens is not None:
                token_ids = token_ids[: int(max_new_tokens)]
            response_tensors.append(
                torch.tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device)
            )

        if not response_tensors:
            response_tensors = [
                torch.tensor([], dtype=input_ids.dtype, device=input_ids.device)
                for _ in range(len(prompts) * num_return_sequences)
            ]

        response_ids = pad_sequence(
            response_tensors,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=response_ids, num_return_sequences=num_return_sequences
        )
        return output

    def start_server(self, data: DataProto, request_complete_callback):
        self.running = True
        while True:
            try:
                command, req_data = self.command_queue.get()
                if command == GenerateRequestType.STOP:
                    break
                if command == GenerateRequestType.ABORT:
                    request_id = req_data.meta_info.get("request_id")
                    if request_id:
                        self.aborted_request_ids.add(request_id)
                    continue
                if command == GenerateRequestType.ALIVE_CHECK:
                    continue
                if command != GenerateRequestType.ADD:
                    continue

                request_id = req_data.meta_info.get("request_id")
                if request_id in self.aborted_request_ids:
                    self.aborted_request_ids.discard(request_id)
                    continue

                generation_config = req_data.meta_info.get("generation_config", {})
                output_token_ids = self._generate_output_token_ids(req_data, generation_config)
                output_data = DataProto(meta_info=req_data.meta_info)
                output_data.meta_info["output_token_ids"] = output_token_ids
                request_complete_callback(data=output_data)
            except Exception as e:
                logger.exception(f"api_infer start_server error: {e}")
                try:
                    if "req_data" in dir() and req_data and "request_id" in req_data.meta_info:
                        output_data = DataProto(meta_info=req_data.meta_info)
                        output_data.meta_info["output_token_ids"] = [[]]
                        request_complete_callback(data=output_data)
                except Exception:
                    pass
                continue

        self.running = False

    def add_request(self, command, data: DataProto):
        self.command_queue.put((command, data))

    def unwrap_model(self):
        return None

    def save_checkpoint(self, *args, **kwargs):
        return {}

    def load_states(self, *args, **kwargs):
        return None

    def offload_states(self, *args, **kwargs):
        return None

    def broadcast_parameter(self, *args, **kwargs):
        return None

    def broadcast_bucket(self, *args, **kwargs):
        return None

    def update_parameter(self, *args, **kwargs):
        return None

    def update_parameter_in_bucket(self, *args, **kwargs):
        return None

    def _decode_prompts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        prompts = []
        if attention_mask is None:
            masks = [None] * input_ids.size(0)
        else:
            masks = attention_mask
        for ids, mask in zip(input_ids, masks):
            if mask is not None:
                ids = ids[mask.bool()]
            prompt = self.tokenizer.decode(ids, skip_special_tokens=self.strip_special_tokens)
            prompts.append(prompt)
        return prompts

    def _complete_prompt(self, prompt: str, num_return_sequences: int, max_new_tokens, temperature, top_p) -> List[str]:
        logger.info(f"API call: model={self.model_name}, prompt_len={len(prompt)}, max_new_tokens={max_new_tokens}")
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries + 1):
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "n": num_return_sequences,
                    "timeout": self.timeout,
                }
                if max_new_tokens is not None:
                    request_kwargs["max_tokens"] = int(max_new_tokens)
                if temperature is not None:
                    request_kwargs["temperature"] = float(temperature)
                if top_p is not None:
                    request_kwargs["top_p"] = float(top_p)
                completion = self.client.chat.completions.create(**request_kwargs)
                outputs = []
                for choice in completion.choices:
                    message = getattr(choice, "message", None)
                    content = ""
                    if message is not None and message.content is not None:
                        content = message.content
                    outputs.append(content)
                if len(outputs) < num_return_sequences:
                    outputs.extend([""] * (num_return_sequences - len(outputs)))
                return outputs
            except Exception:
                if attempt >= self.max_retries:
                    logger.exception("api_infer request failed")
                    raise
                time.sleep(0.5)

        return [""] * num_return_sequences

    def _generate_output_token_ids(self, data: DataProto, generation_config) -> List[List[int]]:
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        prompts = self._decode_prompts(input_ids, attention_mask)

        num_return_sequences = int(generation_config.get("num_return_sequences", 1))
        max_new_tokens = generation_config.get("max_new_tokens") or generation_config.get("max_length")
        do_sample = generation_config.get("do_sample", True)
        temperature = generation_config.get("temperature", 1.0) if do_sample else 0.0
        top_p = generation_config.get("top_p")

        output_token_ids: List[List[int]] = []
        for prompt in prompts:
            responses = self._complete_prompt(
                prompt=prompt,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for text in responses:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                if max_new_tokens is not None:
                    token_ids = token_ids[: int(max_new_tokens)]
                output_token_ids.append(token_ids)
        return output_token_ids
