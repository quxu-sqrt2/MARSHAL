# Multi-Agent memRL Refactor Summary

This document summarizes the Phase 1-4 refactor that converts MARSHAL into a Multi-Agent memRL pipeline: frozen LLM, self-play rollouts, social memory retrieval, and Q-value updates in an external SQLite memory bank.

## Phase 1 - The Freeze (Model Updates Disabled)

What changed
- Added a `freeze_model` switch to the shared config in [roll/configs/base_config.py](roll/configs/base_config.py).
- Gated training-only clusters and updates behind `freeze_model` in [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py).
- Short-circuited `ActorWorker.train_step()` when frozen in [roll/pipeline/base_worker.py](roll/pipeline/base_worker.py).

What it means
- No backpropagation, no optimizer steps, no model update sync.
- Pipeline becomes pure inference + rollout collection.

## Phase 2 - The Memory Engine (SQLite Social Memory Bank)

What changed
- Added a SQLite-backed memory bank with per-player isolation in [roll/agentic/memory/memory_bank.py](roll/agentic/memory/memory_bank.py).
- Added memory configuration knobs in [roll/pipeline/agentic/agentic_config.py](roll/pipeline/agentic/agentic_config.py).

Memory bank behavior
- Two tables: `player_0_bank` and `player_1_bank`.
- `add_experience(intent_emb, experience_text, initial_q)` stores new memories.
- `retrieve_context(intent_emb, top_k1, top_k2)` does Phase-A cosine similarity, then Phase-B Q-value re-rank.
- `update_q_values(ids, reward)` applies $Q_{new} = Q_{old} + \alpha (R - Q_{old})$.

## Phase 3 - Context Hook (Retrieval Injection)

What changed
- Inserted retrieval hook in [roll/agentic/rollout/env_manager.py](roll/agentic/rollout/env_manager.py).
- Built intent text from recent turns, embedded with a mock encoder, then retrieved memory context.
- Injected retrieved experiences into the system prompt.
- Recorded used memory IDs in the rollout metadata.

Result
- LLM generation now sees a social context block before answering.
- Each rollout carries the list of memory IDs it consumed.

## Phase 4 - Utility Update (Social Feedback to Q-values)

What changed
- Added end-of-rollout memory write and Q-value update in [roll/agentic/rollout/env_manager.py](roll/agentic/rollout/env_manager.py).
- Added a `memrl` advantage path that uses response-level Monte Carlo return in [roll/utils/functionals.py](roll/utils/functionals.py).
- Switched the pipeline to `memrl` advantage when `freeze_model` and `memory_enabled` are both true in [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py).

Social feedback logic
- Self-play: compute per-player episode returns, then normalize by role using `normalize_unique_values_by_player`.
- Single-agent: use raw episode return.
- Update Q-values for retrieved memory IDs with the normalized social feedback.
- Add the current trajectory as a new memory entry for the current player.

## Final Pipeline Flow

High-level control flow

+------------------+     +---------------------------+     +---------------------+
| EnvManager       | --> | Intent -> Retrieval       | --> | Prompt Assembly     |
| (state, history) |     | (MemoryBank, SQLite)      |     | (system + user)     |
+------------------+     +---------------------------+     +---------------------+
                                                             |
                                                             v
                                                      +---------------+
                                                      | LLM Generate  |
                                                      +---------------+
                                                             |
                                                             v
+------------------+     +---------------------------+     +----------------------+
| Env Step/Reward  | --> | Rollout Finalize          | --> | Memory Update        |
| (per turn)       |     | (MC return, normalize)    |     | (Q update + add exp) |
+------------------+     +---------------------------+     +----------------------+

Training is disabled in this flow. The only updates are to the external memory bank.

## Key Config Switches

Add these to your Agentic config YAML to enable memRL.

```
freeze_model: true
memory_enabled: true
memory_db_path: ./output/memory/social_memory.sqlite3
memory_top_k1: 50
memory_top_k2: 5
memory_alpha: 0.1
memory_embedding_dim: 128
memory_intent_turns: 3
```

Notes
- `adv_estimator` is automatically set to `memrl` when `freeze_model` and `memory_enabled` are true.
- `memory_db_path` defaults to `output_dir/memory/social_memory.sqlite3` if unset.

## Where to Start (Launch)

Pick one of the existing rollout configs and use the agentic launcher:
- [examples/start_agentic_pipeline.py](examples/start_agentic_pipeline.py)
- Example rollout configs:
  - [examples/hanabi/agentic_rollout_hanabi.yaml](examples/hanabi/agentic_rollout_hanabi.yaml)
  - [examples/tictactoe/agentic_rollout_tictactoe.yaml](examples/tictactoe/agentic_rollout_tictactoe.yaml)
  - [examples/connect_four/agentic_rollout_connect4.yaml](examples/connect_four/agentic_rollout_connect4.yaml)

Typical invocation pattern (replace placeholders with your config folder and YAML stem):

```
python <agentic_launcher> --config_path <config_folder> --config_name <config_stem>
```

What to verify
- SQLite file is created at `memory_db_path`.
- Rollouts contain `memory_ids`, `memory_new_id`, and `memory_updated_ids` in non-tensor metadata.
- Q-values in the memory tables change over episodes.

## Files Touched by Phase

Phase 1
- [roll/configs/base_config.py](roll/configs/base_config.py)
- [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py)
- [roll/pipeline/base_worker.py](roll/pipeline/base_worker.py)

Phase 2
- [roll/agentic/memory/memory_bank.py](roll/agentic/memory/memory_bank.py)
- [roll/pipeline/agentic/agentic_config.py](roll/pipeline/agentic/agentic_config.py)

Phase 3
- [roll/agentic/rollout/env_manager.py](roll/agentic/rollout/env_manager.py)

Phase 4
- [roll/agentic/rollout/env_manager.py](roll/agentic/rollout/env_manager.py)
- [roll/utils/functionals.py](roll/utils/functionals.py)
- [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py)
