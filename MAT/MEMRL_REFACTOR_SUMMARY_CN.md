# MARSHAL → memRL 改造详细指南（中文版）

**文档日期**：2026年4月30日  
**改造版本**：v1.0 - memRL + 模型冻结 + SQLite社会记忆库  
**目标受众**：项目组成员、参与后续实验的学弟学妹

---

## 核心思想：从模型RL到记忆+推理范式

### 为什么要改造？

原始MARSHAL采用**模型参数RL**（PPO/GRPO反向传播）的范式：
```
观察State → LLM生成Action → 环境反馈Reward → 反向传播更新θ → 下一步
```

改造后memRL采用**冻结LLM + 外部记忆RL**的范式：
```
观察State → [查询社会记忆库] → LLM生成Action(带历史提示) → 环境反馈 
                                              ↓
                                          记录体验Experience
                                              ↓
                                    蒙特卡洛Q-value更新
                                      (记忆库更新，不更新θ)
```

**核心差异**：
- ❌ **不再更新LLM参数**（模型完全冻结）
- ✅ **转向更新记忆中的价值估计**（外部Knowledge Base）
- ✅ **推理阶段注入历史最优决策**（Context增强）
- ✅ **通过社会反馈学习角色交互规律**（Multi-agent Learning）

---

## 第一部分：核心memRL机制详解

### 1.1 四大支柱

#### 支柱1️⃣：冻结模型开关（Freeze Model Switch）

**配置参数**：
```yaml
freeze_model: true  # 默认false，保持向后兼容
```

**启用冻结时的效果**：
```
训练初始化阶段：
  ❌ 不初始化 actor_train（训练用的语言模型）
  ❌ 不初始化 critic（价值模型）
  ❌ 不初始化 reference_model（参考模型）
  ✅ 仍保留 actor_inference（纯推理）

训练循环阶段：
  ❌ 不计算 log_probs（对数概率）
  ❌ 不执行 model_update 步骤
  ❌ 不调用任何优化器 step()
  ❌ 不同步参数
  ✅ 仅执行前向推理和数据收集
```

**代码实现位置**：
- [roll/configs/base_config.py](roll/configs/base_config.py) - 参数定义
- [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py) - `__init__` 中的冻结判断
- [roll/pipeline/base_worker.py](roll/pipeline/base_worker.py) - `ActorWorker.train_step()` 的短路逻辑

---

#### 支柱2️⃣：SQLite社会记忆库（Social Memory Bank）

**核心概念**：
- 为每个Player维护一个独立的SQLite表
- 存储该Player在各种游戏State下的历史"成功体验"
- 每条体验关联一个Q-value（社会效用分数）

**数据库架构**：
```
social_memory.sqlite3
├── experiences_p0 表 (Player-0的体验库)
│   ├── id                    INTEGER PRIMARY KEY
│   ├── intent_embedding      BLOB (压缩的np.ndarray)
│   ├── experience_text       TEXT (JSON序列化)
│   ├── q_value               REAL (初始0.0，递增更新)
│   ├── update_count          INTEGER (被用过多少次)
│   └── created_at            TIMESTAMP
│
└── experiences_p1 表 (Player-1的体验库)
    ├── id
    ├── intent_embedding
    ├── experience_text
    ├── q_value
    ├── update_count
    └── created_at
```

**体验JSON结构**：
```json
{
  "intent": "当前棋盘：XXO\nOXX\nO##，我的最近3步动作：[move(0,2), move(1,2), move(0,0)]，当前分数：2",
  "trajectory": [
    {
      "turn": 1,
      "player": 0,
      "action": "place token at (0, 0)",
      "state": "当时的棋盘状态",
      "reward": 0.0
    },
    {
      "turn": 2,
      "player": 1,
      "action": "place token at (1, 1)",
      "state": "对方动作后的棋盘",
      "reward": 0.0
    },
    ...
  ],
  "episode_reward": 1.0,        // Player-0的最终收益
  "team_reward": -1.0,          // Player-1的最终收益
  "game_type": "tic_tac_toe",
  "metadata": {
    "duration_steps": 9,
    "outcome": "win"
  }
}
```

**代码实现**：[roll/agentic/memory/memory_bank.py](roll/agentic/memory/memory_bank.py) (新建)

---

#### 支柱3️⃣：两阶段检索机制（Dual-Stage Retrieval）

**检索流程**：

```
输入：当前Intent（游戏State + 队友历史动作）
                      ↓
        ┌─ Phase-A: 相似度检索 ─────┐
        │                            │
        │ 对所有存储的Experience，  │
        │ 计算Cosine相似度：        │
        │ sim = cos(curr_embed,    │
        │          exp_embed)      │
        │ 取相似度最高的K1条       │
        │ (默认K1=8)              │
        │                            │
        └────────────┬────────────────┘
                     ↓
       ┌─ Phase-B: 价值排序 ──────┐
       │                           │
       │ 从K1条候选中，按Q-value  │
       │ 从高到低排序             │
       │ 取Q-value最高的K2条      │
       │ (默认K2=4)              │
       │ 作为最终检索结果        │
       │                           │
       └─────────────┬─────────────┘
                     ↓
        输出：4条高相似度+高价值的体验
```

**为什么要两阶段？**
- **Phase-A（相似度）**：确保检索的都是"当前游戏状态相近"的历史经验
- **Phase-B（Q值）**：在相似的经验中，优先选择"已被验证为优质"的体验

**配置参数**：
```yaml
memory_top_k_retrieve: [8, 4]    # Phase-A top-8 → Phase-B top-4
```

**代码实现**：[roll/agentic/memory/memory_bank.py](roll/agentic/memory/memory_bank.py) - `retrieve_context()` 方法

---

#### 支柱4️⃣：蒙特卡洛Q值更新（Monte Carlo Q-Value Learning）

**更新公式**：
```
对检索过的每条Experience e：
  Q_new[e] = Q_old[e] + α × (r - Q_old[e])

其中：
  - r = 社会反馈（角色规范化后的Episode Reward）
  - α = 学习率（默认0.1）
  - 关键：只更新本轮检索中用过的体验，不更新未使用的
```

**社会反馈计算**（角色规范化）：
```python
# 针对Self-Play场景
if separate_norm_for_selfplay:
    # Player-0和Player-1分别规范化
    reward_p0_normalized = (episode_reward[0] - mean_p0) / (std_p0 + eps)
    reward_p1_normalized = (episode_reward[1] - mean_p1) / (std_p1 + eps)
    
    social_feedback_p0 = reward_p0_normalized
    social_feedback_p1 = reward_p1_normalized
else:
    # 全局规范化
    combined_rewards = [episode_reward[0], episode_reward[1]]
    normalized = (combined_rewards - mean) / std
    social_feedback_p0 = normalized[0]
    social_feedback_p1 = normalized[1]
```

**为什么规范化？**
- ✅ 避免reward规模差异大时，某个Player的反馈被淹没
- ✅ 确保两个Player对记忆库的贡献权重平衡
- ✅ 在Self-Play中，两个对手的目标相反，分别规范化更合理

**代码实现**：
- [roll/utils/functionals.py](roll/utils/functionals.py) - `memrl` advantage estimator
- [roll/agentic/memory/memory_bank.py](roll/agentic/memory/memory_bank.py) - `update_q_values()` 方法

---

### 1.2 Intent（查询意图）的构建

**Intent是什么？**
Intent是当前游戏State的结构化表示，用于与记忆库中的历史体验匹配。

**Intent构建规则**：
```
Intent = Concatenate([
    "Current Game State:\n" + str(game_state),
    "\nRecent Teammate Actions:\n" + str(teammate_history[-N:]),
    "\nCurrent Score:\n" + str(current_score)
])

其中：
  - game_state: 当前的棋盘/环境状态（字符串表示）
  - teammate_history: 队友最近N步的动作序列（N默认为5）
  - current_score: 当前累积分数
```

**示例（Tic-Tac-Toe）**：
```
Current Game State:
X O X
O X .
. . .

Recent Teammate Actions:
['place(0,1)', 'place(1,1)']

Current Score:
2
```

**Intent嵌入的生成**：
```
1. 将Intent字符串哈希：hash_value = hashlib.sha256(intent_str)
2. 用哈希值初始化RNG种子
3. 随机生成256维的嵌入向量：embed = rng.randn(256) * scale
4. 存储到SQLite的intent_embedding字段（经zlib压缩）
```

**为什么用哈希？**
- ✅ 相同的State会生成相同的嵌入，便于匹配
- ✅ 快速、确定性、无需训练
- ✅ 可后续替换为真实的embedding模型（BERT等）

---

## 第二部分：完整数据流程图

### 2.1 生成阶段（Inference Phase）

```
┌──────────────────────────────────────────────────────────────────┐
│                     STEP 1: 生成阶段                              │
│                  (Inference & Context Injection)                  │
└──────────────────────────────────────────────────────────────────┘

env.reset() / env.step(prev_action)
         ↓
    env_output = {
      "state": "当前棋盘: X O X / O . . / . . .",
      "legal_actions": [移动1, 移动2, 移动3],
      "current_player": 0,
      "rewards": [0, 0]
    }
         ↓
    ┌─────────────────────────────────────────────────────┐
    │ 构建当前Intent                                       │
    ├─────────────────────────────────────────────────────┤
    │ current_intent = extract_intent(               │
    │   state=env_output["state"],                        │
    │   teammate_history=[上一步对手的动作],                │
    │   current_score=累积分数                             │
    │ )                                                    │
    │                                                      │
    │ current_intent_embedding = hash_embed(intent)       │
    └──────────────┬──────────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────────┐
    │ IF memory_enabled AND current_player in memory_bank:│
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │ retrieved_exps = memory_bank[current_player]        │
    │   .retrieve_context(                                │
    │     intent_embedding=current_intent_embedding,      │
    │     player_id=current_player                        │
    │   )                                                  │
    │                                                      │
    │ 返回：List[Experience]，长度≤4                      │
    │                                                      │
    │ ✅ 同时记录检索到的记忆ID：                          │
    │    memory_retrieved_ids[current_player] = [id1, id2, ...]
    │                                                      │
    └──────────────┬──────────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────────┐
    │ 构建Context提示文本                                   │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │ IF retrieved_exps is not empty:                      │
    │   context_str = "## Historical Successful Patterns:\n"
    │   for exp in retrieved_exps:                        │
    │     context_str += f"- Experience {exp.id}:\n"      │
    │     context_str += f"  State: {exp.state}\n"        │
    │     context_str += f"  Actions: {exp.actions}\n"    │
    │     context_str += f"  Q-Value: {exp.q_value:.3f}\n"
    │                                                      │
    │ ELSE:                                                │
    │   context_str = "## No Historical Patterns Found.\n"
    │                                                      │
    └──────────────┬──────────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────────┐
    │ 构建LLM输入Prompt                                    │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │ system_prompt = (系统提示词)                          │
    │ context_prompt = context_str                        │
    │ turn_prompt = f"Current state: {env_output['state']}"
    │             + f"Legal actions: {legal_actions}"      │
    │                                                      │
    │ full_prompt = system_prompt + context_prompt        │
    │             + turn_prompt                           │
    │                                                      │
    │ messages = [                                        │
    │   {"role": "system", "content": full_prompt},       │
    │   {"role": "user", "content": turn_prompt}          │
    │ ]                                                   │
    │                                                      │
    └──────────────┬──────────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────────┐
    │ 🔒 LLM推理（模型冻结，仅前向传播）                    │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │ llm_input = tokenizer.encode(messages)              │
    │ llm_output_ids = llm.generate(                       │
    │   input_ids=llm_input,                              │
    │   max_new_tokens=256,                               │
    │   temperature=1.0                                   │
    │ )                                                   │
    │                                                      │
    │ ⚠️  NO backward pass, NO gradient update            │
    │ ⚠️  NO parameter synchronization                    │
    │                                                      │
    └──────────────┬──────────────────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────────────────┐
    │ 解析LLM输出为动作                                     │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │ action_text = tokenizer.decode(llm_output_ids)      │
    │ action = parse_action(action_text,                  │
    │                      legal_actions)                 │
    │                                                      │
    │ 📝 记录Action用过的记忆ID：                          │
    │    DataProto.meta_info['memory_ids'] =             │
    │      memory_retrieved_ids[current_player]           │
    │                                                      │
    └──────────────┬──────────────────────────────────────┘
                   ↓
          返回action进行环境step()
          继续下一轮或转入环境执行阶段
```

---

### 2.2 环境执行阶段（Environment Phase）

```
┌──────────────────────────────────────────────────────────────────┐
│                  STEP 2: 环境执行 & 轨迹收集                       │
│                    (Environment Rollout)                           │
└──────────────────────────────────────────────────────────────────┘

action（来自STEP 1）
    ↓
env.step(action)
    ↓
result = {
  "next_state": "更新后的棋盘",
  "rewards": [0, 0],        # [Player-0奖励, Player-1奖励]
  "done": False,            # 是否游戏结束
  "info": {}
}
    ↓
    ┌──────────────────────────────────────┐
    │ 更新轨迹记录                           │
    ├──────────────────────────────────────┤
    │ trajectory.append({                  │
    │   "turn": current_step,              │
    │   "player": current_player,          │
    │   "action": action,                  │
    │   "state": next_state,               │
    │   "reward": rewards[current_player]  │
    │ })                                   │
    │                                      │
    │ cumulative_rewards[0] += rewards[0]  │
    │ cumulative_rewards[1] += rewards[1]  │
    └─────────────────┬────────────────────┘
                      ↓
    ┌─────────────────────────────────────┐
    │ IF done == True:                    │
    │   → 转入STEP 3（回合结算）             │
    │ ELSE:                               │
    │   → 回到STEP 1继续生成                │
    └─────────────────────────────────────┘
```

---

### 2.3 回合结算阶段（Episode Finalization Phase）

```
┌──────────────────────────────────────────────────────────────────┐
│               STEP 3: 回合结算 & memRL更新                         │
│            (Episode Termination & Memory Update)                   │
└──────────────────────────────────────────────────────────────────┘

游戏结束，得到：
  - final_state: 最终棋盘状态
  - trajectory: [(turn, player, action, state, reward), ...]
  - cumulative_rewards = [R_p0, R_p1]
    ↓
    ┌──────────────────────────────────────────────────────┐
    │ 为每个Player生成体验Experience                        │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ for player_id in [0, 1]:                             │
    │   # 构建该Player的Intent                              │
    │   intent_p = extract_intent(                         │
    │     state=final_state,                               │
    │     teammate_history=player_actions[1-player_id],   │
    │     current_score=...                                │
    │   )                                                  │
    │   intent_embed_p = hash_embed(intent_p)              │
    │                                                       │
    │   # 构建该Player的体验JSON                            │
    │   experience_json = {                                │
    │     "intent": intent_p,                              │
    │     "trajectory": trajectory,                        │
    │     "episode_reward": cumulative_rewards[player_id], │
    │     "team_reward": cumulative_rewards[1-player_id],  │
    │     "game_type": game_type,                          │
    │     "metadata": {                                    │
    │       "duration_steps": len(trajectory),             │
    │       "outcome": determine_outcome(player_id)        │
    │     }                                                │
    │   }                                                  │
    │                                                       │
    │   # ✅ 写入记忆库                                      │
    │   new_exp_id_p = memory_bank[player_id]             │
    │     .add_experience(                                 │
    │       intent_embedding=intent_embed_p,               │
    │       experience_text=json.dumps(experience_json),   │
    │       initial_q=0.0                                  │
    │     )                                                │
    │                                                       │
    │   📊 记录新体验ID：                                    │
    │   memory_new_ids.append(new_exp_id_p)                │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────────────┐
    │ 计算社会反馈（角色规范化的奖励）                        │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ # 使用全局或分离的规范化                               │
    │ IF reward_normalization.separate_norm_for_selfplay: │
    │   # Player-0和Player-1分别规范化                      │
    │   raw_rewards_p0 = [所有历史Episode中Player-0的奖励] │
    │   raw_rewards_p1 = [所有历史Episode中Player-1的奖励] │
    │                                                       │
    │   mean_p0 = np.mean(raw_rewards_p0)                  │
    │   std_p0 = np.std(raw_rewards_p0)                    │
    │   mean_p1 = np.mean(raw_rewards_p1)                  │
    │   std_p1 = np.std(raw_rewards_p1)                    │
    │                                                       │
    │   social_fb_p0 = (cumulative_rewards[0] - mean_p0)  │
    │                   / (std_p0 + 1e-8)                  │
    │   social_fb_p1 = (cumulative_rewards[1] - mean_p1)  │
    │                   / (std_p1 + 1e-8)                  │
    │                                                       │
    │ ELSE:                                                │
    │   # 全局规范化                                        │
    │   combined = [cumulative_rewards[0],                 │
    │              cumulative_rewards[1]]                  │
    │   mean = np.mean(combined)                           │
    │   std = np.std(combined)                             │
    │   social_fb = (combined - mean) / (std + 1e-8)       │
    │   social_fb_p0 = social_fb[0]                        │
    │   social_fb_p1 = social_fb[1]                        │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────────────┐
    │ 🔄 更新检索过的体验的Q-value                            │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ for player_id in [0, 1]:                             │
    │   retrieved_ids = memory_retrieved_ids[player_id]   │
    │   social_feedback = social_fb_p0 or social_fb_p1    │
    │                                                       │
    │   if retrieved_ids is not empty:  # 仅更新用过的      │
    │     memory_bank[player_id].update_q_values(          │
    │       memory_ids=retrieved_ids,                      │
    │       reward=social_feedback,                        │
    │       alpha=0.1  # 学习率                            │
    │     )                                                │
    │                                                       │
    │     # 伪代码：对每条检索过的体验做Q-update           │
    │     for exp_id in retrieved_ids:                     │
    │       exp = memory_bank[player_id].get(exp_id)       │
    │       Q_new = exp.q_value + 0.1 * (social_feedback  │
    │                                     - exp.q_value)   │
    │       exp.q_value = Q_new                            │
    │       exp.update_count += 1                          │
    │       memory_bank[player_id].commit()  # 保存到DB    │
    │                                                       │
    │   📊 记录更新的ID：                                    │
    │   memory_updated_ids.append(retrieved_ids)           │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────────────┐
    │ 📈 记录关键指标                                         │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ dump_data = {                                        │
    │   "memories": {                                      │
    │     "new_ids": memory_new_ids,                       │
    │     "updated_ids": memory_updated_ids,               │
    │     "social_feedback": [social_fb_p0, social_fb_p1],│
    │     "num_retrieved_p0": len(memory_retrieved_ids[0]),│
    │     "num_retrieved_p1": len(memory_retrieved_ids[1]),│
    │     "avg_q_value_after": [...]                       │
    │   },                                                 │
    │   "trajectory": trajectory,                          │
    │   "rewards": cumulative_rewards                      │
    │ }                                                    │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
          进入STEP 4（模型优化，仅当freeze_model=False）
```

---

### 2.4 模型优化阶段（Model Update Phase）

```
┌──────────────────────────────────────────────────────────────────┐
│          STEP 4: 模型优化 & 梯度计算                              │
│     (仅当 freeze_model == False 时执行)                           │
└──────────────────────────────────────────────────────────────────┘

IF freeze_model == False:
    ↓
    ┌──────────────────────────────────────────────────────┐
    │ 计算优势估计（Advantage Estimation）                  │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ IF freeze_model == False AND memory_enabled == True: │
    │   # 使用memrl advantage估计                            │
    │   adv_estimator = "memrl"                            │
    │   # mc_return = sum(episode_rewards)                 │
    │   # advantages = broadcast to response length        │
    │                                                       │
    │ ELIF freeze_model == False AND memory_enabled == False
    │   # 使用传统的GAE或其他                               │
    │   adv_estimator = "gae"  # 或 reinforce 等           │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────────────┐
    │ 🔃 批量数据处理                                        │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ 收集多个Episode的数据 → 形成batch                     │
    │ batch = {                                            │
    │   "input_ids": [prompt_ids_1, prompt_ids_2, ...],   │
    │   "response_ids": [resp_ids_1, resp_ids_2, ...],    │
    │   "rewards": [R_1, R_2, ...],                        │
    │   "advantages": [adv_1, adv_2, ...],                 │
    │   ...                                                │
    │ }                                                    │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────────────────┐
    │ 🚀 梯度计算与参数更新                                  │
    ├──────────────────────────────────────────────────────┤
    │                                                       │
    │ actor_train.train_step(batch)  # 前向 + 反向          │
    │   ├─ log_probs = actor(input, response)              │
    │   ├─ loss = policy_loss(log_probs, advantages)       │
    │   ├─ loss.backward()  # 反向传播                      │
    │   └─ optimizer.step()  # 参数更新                     │
    │                                                       │
    │ reference_model.sync()  # 参考模型同步                │
    │ critic.train_step(batch)  # Critic更新（如果使用GAE） │
    │                                                       │
    └─────────────────────┬──────────────────────────────┘
                          ↓
          ✅ 模型参数更新完成
          准备下一轮Rollout

IF freeze_model == True:
    ↓
    ✅ 跳过所有反向传播
    ✅ 仅输出指标、记忆更新、轨迹
    ✅ 模型参数完全冻结
```

---

## 第三部分：代码改动清单

### 3.1 Phase 1: 模型冻结（Freeze Model）

**关键文件**：
1. [roll/configs/base_config.py](roll/configs/base_config.py)
2. [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py)
3. [roll/pipeline/base_worker.py](roll/pipeline/base_worker.py)

**改动内容**：
```python
# roll/configs/base_config.py
@dataclass
class BaseConfig:
    ...
    freeze_model: bool = False  # 🔒 新增开关，默认false保持兼容
    ...
```

```python
# roll/pipeline/agentic/agentic_pipeline.py
class AgenticPipeline(BasePipeline):
    def __init__(self, ...):
        ...
        if not self.config.freeze_model:
            # 初始化训练相关的模型
            self.actor_train = init_actor_train(...)
            self.critic = init_critic(...)
            self.reference_model = init_reference(...)
        else:
            # 冻结时仅保留推理模型
            self.actor_train = None
            self.critic = None
            self.reference_model = None
        ...

    def run(self, ...):
        ...
        if not self.config.freeze_model:
            # 正常的PPO/GRPO循环
            loss = model_update(...)
            actor_train.train_step(loss)
        else:
            # 冻结模式：仅收集数据，不更新参数
            log_probs = None  # 不计算
            advantages = compute_reinforce_advantages(...)  # 降级为REINFORCE
        ...
```

```python
# roll/pipeline/base_worker.py
class ActorWorker:
    def train_step(self, batch):
        if self.config.freeze_model:
            # 短路返回，不执行任何训练操作
            return {
                "loss": 0.0,
                "log_probs": None
            }
        else:
            # 正常的训练逻辑
            ...
```

---

### 3.2 Phase 2+3: SQLite记忆库 & 检索注入

**关键文件**：
1. [roll/agentic/memory/memory_bank.py](roll/agentic/memory/memory_bank.py) (新建)
2. [roll/pipeline/agentic/agentic_config.py](roll/pipeline/agentic/agentic_config.py)
3. [roll/agentic/rollout/env_manager.py](roll/agentic/rollout/env_manager.py)

**memory_bank.py 核心接口**：
```python
class SocialMemoryBank:
    """为单个Player维护一个SQLite表的体验库"""
    
    def __init__(self, db_path: str, player_id: int, 
                 embedding_dim: int = 256):
        """初始化数据库连接和表结构"""
        pass
    
    def add_experience(self, intent_embedding: np.ndarray,
                       experience_text: str,
                       initial_q: float = 0.0) -> int:
        """
        添加新体验到记忆库
        
        返回：新体验的ID
        """
        pass
    
    def retrieve_context(self, intent_embedding: np.ndarray,
                         top_k_list: List[int] = [8, 4]) \
                    -> List[Experience]:
        """
        两阶段检索：先相似度→再Q值排序
        
        top_k_list[0]: Phase-A的top-k
        top_k_list[1]: Phase-B的top-k
        """
        pass
    
    def update_q_values(self, memory_ids: List[int],
                        reward: float,
                        alpha: float = 0.1) -> None:
        """
        蒙特卡洛更新Q值：Q_new = Q_old + alpha * (reward - Q_old)
        """
        pass
```

**env_manager.py 的改动**：
```python
class EnvManager:
    def __init__(self, config):
        ...
        if config.memory_enabled:
            # 为每个Player初始化记忆库
            self.memory_banks = {
                player_id: SocialMemoryBank(
                    db_path=config.memory_db_path,
                    player_id=player_id,
                    embedding_dim=config.memory_embedding_dim
                )
                for player_id in range(num_players)
            }
            self.memory_retrieved_ids = {player_id: [] 
                                        for player_id in range(num_players)}
        else:
            self.memory_banks = None
        ...

    def _format_messages(self, state, player_id, ...):
        """在生成前注入检索到的历史体验作为Context"""
        
        # 提取Intent
        intent = self._extract_intent(state, player_id)
        intent_embedding = self._hash_embed(intent)
        
        # 检索记忆
        retrieved_exps = []
        if self.memory_banks and player_id in self.memory_banks:
            retrieved_exps = self.memory_banks[player_id]\
                .retrieve_context(intent_embedding)
            # 记录检索ID用于后续更新
            self.memory_retrieved_ids[player_id] = \
                [exp.id for exp in retrieved_exps]
        
        # 构建Context文本
        context_str = self._format_context(retrieved_exps)
        
        # 拼接messages
        messages = [
            {
                "role": "system",
                "content": base_prompt + context_str
            },
            {
                "role": "user",
                "content": f"State: {state}\nLegal actions: ..."
            }
        ]
        
        return messages, {
            "memory_ids": self.memory_retrieved_ids[player_id],
            "retrieved_count": len(retrieved_exps)
        }

    def _formulate_single_rollout(self, trajectory, rewards, ...):
        """在轨迹完成后，写入新体验并更新Q值"""
        
        # 计算社会反馈（角色规范化）
        social_feedback_p0 = normalize_reward(rewards[0], player_id=0)
        social_feedback_p1 = normalize_reward(rewards[1], player_id=1)
        
        # 为每个Player写入新体验
        new_ids = []
        for player_id in [0, 1]:
            intent_p = self._extract_intent(final_state, player_id)
            experience_json = {
                "intent": intent_p,
                "trajectory": trajectory,
                "episode_reward": rewards[player_id],
                "team_reward": rewards[1-player_id]
            }
            
            new_id = self.memory_banks[player_id]\
                .add_experience(
                    intent_embedding=self._hash_embed(intent_p),
                    experience_text=json.dumps(experience_json),
                    initial_q=0.0
                )
            new_ids.append(new_id)
        
        # 🔄 更新检索过的体验的Q值
        for player_id in [0, 1]:
            retrieved_ids = self.memory_retrieved_ids[player_id]
            social_fb = social_feedback_p0 if player_id == 0 \
                       else social_feedback_p1
            
            if retrieved_ids:
                self.memory_banks[player_id]\
                    .update_q_values(
                        memory_ids=retrieved_ids,
                        reward=social_fb,
                        alpha=0.1
                    )
        
        return {
            "memory_new_ids": new_ids,
            "memory_updated_ids": [
                self.memory_retrieved_ids[0],
                self.memory_retrieved_ids[1]
            ]
        }
```

---

### 3.3 Phase 4: 蒙特卡洛优势估计 & 角色规范化

**关键文件**：
1. [roll/utils/functionals.py](roll/utils/functionals.py)
2. [roll/pipeline/agentic/agentic_pipeline.py](roll/pipeline/agentic/agentic_pipeline.py)

**functionals.py 新增**：
```python
def compute_memrl_advantages(rewards, response_lengths, ...):
    """
    memRL模式的优势估计（蒙特卡洛）
    
    逻辑：
      1. 对每个Episode计算累积Return
      2. 规范化该Return（针对角色）
      3. 广播到Response的每个Token
    """
    
    mc_returns = []
    for r in rewards:
        mc_return = sum(r)  # 累积回报
        mc_returns.append(mc_return)
    
    # 角色规范化
    normalized_returns = normalize_unique_values_by_player(
        values=mc_returns,
        player_ids=[0, 1],  # self-play
        separate_norm=True
    )
    
    # 广播到response长度
    advantages = []
    for norm_ret, resp_len in zip(normalized_returns, response_lengths):
        advantages.append([norm_ret] * resp_len)
    
    return advantages


def normalize_unique_values_by_player(values, player_ids, separate_norm=True):
    """
    按Player分别规范化奖励
    
    参数：
      values: [v0, v1] - Player-0和Player-1的奖励
      player_ids: [0, 1]
      separate_norm: True时分别规范化，False时全局规范化
    """
    
    if separate_norm:
        # Player-0: 收集所有历史Episode中Player-0的奖励，计算mean/std
        hist_vals_p0 = historical_rewards[0]  # 全局历史记录
        mean_p0 = np.mean(hist_vals_p0)
        std_p0 = np.std(hist_vals_p0)
        norm_v0 = (values[0] - mean_p0) / (std_p0 + 1e-8)
        
        # Player-1: 同理
        hist_vals_p1 = historical_rewards[1]
        mean_p1 = np.mean(hist_vals_p1)
        std_p1 = np.std(hist_vals_p1)
        norm_v1 = (values[1] - mean_p1) / (std_p1 + 1e-8)
        
        return [norm_v0, norm_v1]
    else:
        # 全局规范化
        combined = list(values)
        mean = np.mean(combined)
        std = np.std(combined)
        normalized = [(v - mean) / (std + 1e-8) for v in combined]
        return normalized
```

**agentic_pipeline.py 的改动**：
```python
class AgenticPipeline(BasePipeline):
    def run(self, ...):
        ...
        
        # 选择优势估计方法
        if self.config.freeze_model and self.config.memory_enabled:
            # memRL模式：使用蒙特卡洛
            advantages = compute_memrl_advantages(
                rewards=batch_rewards,
                response_lengths=batch_response_lens
            )
        else:
            # 原始模式：使用GAE等
            advantages = compute_gae_advantages(...)
        
        # 后续处理
        ...
```

---

## 第四部分：启动与配置

### 4.1 基础配置示例（Tic-Tac-Toe）

**文件**：`examples/tictactoe/config/agentic_tictactoe_memrl.yaml`

```yaml
# ============= 基础设置 =============
output_dir: ./output/tictactoe_memrl_v1
seed: 42
num_rollouts: 100
rollout_workers: 4

# ============= 模型配置 =============
model_name: qwen-2.5-7b-instruct
actor_lora_rank: 64
critic_lora_rank: 64

# 🔒 冻结模型开关（核心改动）
freeze_model: true

# ============= memRL 记忆库配置 =============
memory_enabled: true
memory_db_path: ${output_dir}/memory/social_memory.sqlite3
memory_embedding_dim: 256
memory_intent_window: 5  # 构建Intent时回看最近5步队友动作

# 两阶段检索参数
memory_top_k_retrieve: [8, 4]  # Phase-A top-8 → Phase-B top-4
memory_learning_rate: 0.1      # Q-value学习率

# ============= 优势估计 =============
adv_estimator: memrl  # 冻结+记忆启用时自动使用

# ============= 奖励规范化 =============
reward_normalization:
  method: mean_std
  separate_norm_for_selfplay: true  # Player-0/1分别规范化

# ============= 数据收集 =============
collect_mode: train
max_steps_per_episode: 100
num_episodes: 500

# ============= 日志与监控 =============
log_dir: ${output_dir}/logs
log_interval: 10
save_interval: 50
```

---

### 4.2 启动命令

```bash
# 进入项目目录
cd /path/to/MARSHAL

# 方式1：使用配置文件启动
python examples/start_agentic_pipeline.py \
  --config_path examples/tictactoe/config \
  --config_name agentic_tictactoe_memrl \
  --num_rollouts 200

# 方式2：指定输出目录
python examples/start_agentic_pipeline.py \
  --config_path examples/tictactoe/config \
  --config_name agentic_tictactoe_memrl \
  --output_dir ./results/exp_memrl_v1

# 方式3：调试模式（单进程）
python examples/start_agentic_pipeline.py \
  --config_path examples/tictactoe/config \
  --config_name agentic_tictactoe_memrl \
  --rollout_workers 1 \
  --debug
```

---

### 4.3 查询和分析记忆库

```python
import sqlite3
import json
import numpy as np

# 连接数据库
db_path = "./output/tictactoe_memrl_v1/memory/social_memory.sqlite3"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# ========== 查看Player-0的体验统计 ==========
cur.execute("SELECT COUNT(*) FROM experiences_p0")
total_exp_p0 = cur.fetchone()[0]
print(f"Player-0 总体验数: {total_exp_p0}")

# 查看Q-value分布
cur.execute("""
    SELECT 
        COUNT(*) as count,
        ROUND(AVG(q_value), 3) as avg_q,
        ROUND(MIN(q_value), 3) as min_q,
        ROUND(MAX(q_value), 3) as max_q
    FROM experiences_p0
""")
stats = cur.fetchone()
print(f"  平均Q-value: {stats[1]}")
print(f"  最小Q-value: {stats[2]}")
print(f"  最大Q-value: {stats[3]}")

# ========== 查看更新最频繁的体验 ==========
cur.execute("""
    SELECT id, q_value, update_count 
    FROM experiences_p0 
    ORDER BY update_count DESC 
    LIMIT 5
""")
print("\n更新最频繁的5条体验（Player-0）:")
for row in cur.fetchall():
    exp_id, q_val, update_cnt = row
    print(f"  ID={exp_id}, Q={q_val:.3f}, 更新次数={update_cnt}")

# ========== 查看高Q-value的体验 ==========
cur.execute("""
    SELECT id, q_value, update_count, experience_text 
    FROM experiences_p0 
    WHERE q_value > 0.5
    ORDER BY q_value DESC 
    LIMIT 3
""")
print("\n高Q-value的体验（Player-0）:")
for row in cur.fetchall():
    exp_id, q_val, update_cnt, exp_json_str = row
    exp_data = json.loads(exp_json_str)
    print(f"  ID={exp_id}, Q={q_val:.3f}, 更新次数={update_cnt}")
    print(f"    Intent: {exp_data['intent'][:100]}...")
    print(f"    Outcome: {exp_data['metadata']['outcome']}")
    print()

# ========== 比较两个Player的Q-value分布 ==========
cur.execute("SELECT AVG(q_value), STDDEV(q_value) FROM experiences_p0")
mean_p0, std_p0 = cur.fetchone()

cur.execute("SELECT AVG(q_value), STDDEV(q_value) FROM experiences_p1")
mean_p1, std_p1 = cur.fetchone()

print(f"\n两个Player的Q-value分布:")
print(f"  Player-0: 均值={mean_p0:.3f}, 标差={std_p0:.3f}")
print(f"  Player-1: 均值={mean_p1:.3f}, 标差={std_p1:.3f}")

conn.close()
```

---

### 4.4 监控训练进度

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
log_file = "./output/tictactoe_memrl_v1/logs/training.jsonl"
records = []
with open(log_file, 'r') as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# 绘制学习曲线
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Episode Reward
axes[0, 0].plot(df['episode_num'], df['episode_reward_p0'], label='Player-0')
axes[0, 0].plot(df['episode_num'], df['episode_reward_p1'], label='Player-1')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('Episode Reward Over Time')
axes[0, 0].legend()
axes[0, 0].grid()

# 2. Average Q-value
axes[0, 1].plot(df['episode_num'], df['avg_q_value_p0'], label='Player-0')
axes[0, 1].plot(df['episode_num'], df['avg_q_value_p1'], label='Player-1')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Avg Q-value')
axes[0, 1].set_title('Average Q-value Trend')
axes[0, 1].legend()
axes[0, 1].grid()

# 3. 记忆库大小
axes[1, 0].plot(df['episode_num'], df['total_memories_p0'], label='Player-0')
axes[1, 0].plot(df['episode_num'], df['total_memories_p1'], label='Player-1')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Total Memories')
axes[1, 0].set_title('Memory Bank Size')
axes[1, 0].legend()
axes[1, 0].grid()

# 4. 检索频率
axes[1, 1].plot(df['episode_num'], df['retrieved_count_p0'], label='Player-0')
axes[1, 1].plot(df['episode_num'], df['retrieved_count_p1'], label='Player-1')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Retrieved Count')
axes[1, 1].set_title('Memory Retrieval Frequency')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.savefig('./output/tictactoe_memrl_v1/training_curves.png', dpi=150)
print("✅ 学习曲线已保存到 training_curves.png")
```

---

## 第五部分：核心概念对比与设计理由

### 5.1 为什么要冻结模型？

| 维度 | 原始RL（PPO/GRPO） | memRL冻结模式 |
|------|------------------|-------------|
| **学习速度** | 🐌 慢（反向传播成本大） | ⚡ 快（仅推理+查表） |
| **内存占用** | 📦 大（需存梯度+优化器状态） | 💾 小（仅推理模型） |
| **并行扩展** | 📊 受限（梯度同步） | 🚀 高（无同步需求） |
| **可解释性** | ❓ 黑盒（参数更新） | 📖 白盒（记忆查询） |
| **知识迁移** | 🔄 模型编码（难以调整） | 📚 显式记忆（易审计） |
| **多智能体** | ⚠️ 难以隔离（混合梯度） | ✅ 角色隔离（独立Q值） |

**结论**：冻结模型适合需要**快速推理、可解释性、多智能体交互**的场景。

---

### 5.2 为什么要SQLite而不是向量数据库？

| 选项 | SQLite | Faiss/Milvus |
|------|--------|-------------|
| **部署成本** | ✅ 零依赖，文件存储 | ⚠️ 需单独服务 |
| **查询灵活性** | ✅ SQL支持复杂条件 | ❓ 主要是向量查询 |
| **更新效率** | ✅ 事务支持，一致性强 | ⚠️ 索引重建成本 |
| **持久化** | ✅ 原生支持 | 🔄 需外部存储 |
| **开发难度** | ✅ 极低 | ⚠️ 较高 |

**结论**：对于prototype和研究，SQLite是最快的方案。生产环境可迁移到向量数据库。

---

### 5.3 为什么要两阶段检索？

**单阶段（仅相似度）的问题**：
- ❌ 高相似度的体验不一定高价值
- ❌ 可能检索到"相似但失败"的体验
- ❌ 没有利用历史Q值信息

**两阶段的优势**：
- ✅ Phase-A确保**语义相关性**
- ✅ Phase-B确保**价值最高**
- ✅ 结合了**相似性**和**质量**两个维度

**时间成本**：
```
单阶段：O(N) - N为总体验数
两阶段：O(N) + O(K1 log K1) ≈ O(N) - K1通常很小
```

---

### 5.4 角色规范化的数学原理

**Self-Play中的奖励问题**：
```
Episode结束：
  Player-0 获得 reward=1.0（赢）
  Player-1 获得 reward=-1.0（输）

直接规范化会导致：
  - Player-0和Player-1的梯度方向完全相反
  - 两个Player对记忆库的更新互相抵消
```

**分离规范化的解决方案**：
```
维护两个独立的统计量：
  - mean_p0, std_p0 = statistics of Player-0's historical rewards
  - mean_p1, std_p1 = statistics of Player-1's historical rewards

对每个Player分别规范化：
  feedback_p0 = (reward_p0 - mean_p0) / std_p0
  feedback_p1 = (reward_p1 - mean_p1) / std_p1

结果：
  - 都相对于自己的历史分布规范化
  - 消除了奖励规模的不对称性
  - 两个Player可独立学习
```

**实例**：
```
历史数据：
  Player-0的奖励：[1.0, 1.0, 0.0, -1.0] → mean=0.25, std=0.83
  Player-1的奖励：[-1.0, -1.0, 0.0, 1.0] → mean=-0.25, std=0.83

本轮Episode：
  Player-0获得 reward=1.0
  Player-1获得 reward=-1.0

规范化反馈：
  feedback_p0 = (1.0 - 0.25) / 0.83 = +0.90
  feedback_p1 = (-1.0 - (-0.25)) / 0.83 = -0.90

💡 结果：两者的反馈大小相同（都是0.90），但方向相反
💡 这样Player-0和Player-1对记忆的贡献是平衡的
```

---

## 第六部分：常见问题与故障排查

### 6.1 Q&A

**Q: 为什么新体验初始化Q=0而不是episode_reward？**

A: 为了建立**因果隔离**。新体验的真实价值需要通过多轮检索和反馈来逐步发现，不应一开始就高估。

**Q: 记忆库会无限增长吗？**

A: 是的。生产环境可添加：
```python
if len(memory_bank) > MAX_SIZE:
    # 删除Q值最低的1%的体验
    del_count = max_size // 100
    low_q_ids = memory_bank.get_lowest_q_ids(del_count)
    memory_bank.delete_by_ids(low_q_ids)
```

**Q: Intent嵌入用哈希函数是否太简陋？**

A: 目前是Prototype。可升级为：
- 句子BERT（ST-BERT）
- 游戏专用的CNN嵌入
- 小型Transformer编码器
代码框架已预留了接口。

**Q: 多进程并发写入记忆库会冲突吗？**

A: SQLite默认有文件锁保护。建议：
```yaml
memory_db_config:
  timeout: 30  # 等待锁的超时时间（秒）
  journal_mode: WAL  # Write-Ahead Logging，提高并发性
```

**Q: 内存占用如何评估？**

A: 单条体验约：
- 文本JSON: 200-500字节
- 压缩嵌入: 1KB（256维）
- 元数据: 100字节
**总计**：~1.5KB/体验

1000条体验 ≈ 1.5MB，不会有问题。

---

### 6.2 故障排查表

| 症状 | 原因 | 解决方案 |
|------|------|--------|
| Q-value始终为0 | `memory_retrieved_ids`为空 | 检查是否 `mode=="train"` 且Memory启用 |
| `database is locked` | 多进程竞争 | 增加 `timeout` 或切换 `journal_mode: WAL` |
| 推理延迟大幅增加 | 记忆库查询变慢 | 添加索引：`CREATE INDEX ON experiences_pX(q_value)` |
| Context过长导致截断 | 检索条数太多 | 减少 `memory_top_k_retrieve[1]` |
| 模型仍在更新 | `freeze_model=False` | 确认配置文件中 `freeze_model: true` |
| Intent相似度全为0 | 嵌入维度不匹配 | 检查 `memory_embedding_dim` 配置一致性 |

---

### 6.3 性能调优建议

**快速检索**：
```yaml
# 添加索引
PRAGMA foreign_keys = ON;
CREATE INDEX idx_q_value_p0 ON experiences_p0(q_value DESC);
CREATE INDEX idx_q_value_p1 ON experiences_p1(q_value DESC);
```

**减少内存占用**：
```yaml
memory_top_k_retrieve: [4, 2]  # 从[8,4]降低到[4,2]
memory_intent_window: 3         # 从5降低到3
```

**加速记忆库初始化**：
```python
# 预加载所有embedding到内存
class SocialMemoryBankOptimized(SocialMemoryBank):
    def __init__(self, ...):
        super().__init__(...)
        self.embedding_cache = {}  # 缓存所有embedding
        self._load_all_embeddings()
```

---

## 第七部分：进阶扩展

### 7.1 集成真实Embedding模型

```python
from sentence_transformers import SentenceTransformer

class SocialMemoryBankWithRealEmbedding(SocialMemoryBank):
    def __init__(self, db_path, player_id, model_name="all-MiniLM-L6-v2"):
        super().__init__(db_path, player_id)
        self.embedding_model = SentenceTransformer(model_name)
    
    def _compute_embedding(self, intent_text: str) -> np.ndarray:
        """使用句子BERT生成嵌入"""
        embedding = self.embedding_model.encode(intent_text)
        return embedding
```

---

### 7.2 与传统RL混合（Hybrid Mode）

```python
# Phase 1-2: 冻结模式收集体验
freeze_model: true
memory_enabled: true
num_episodes: 500

# Phase 3: 基于记忆的Policy Distillation
# 不直接反向传播，而是蒸馏记忆库中的知识到新模型

def distill_policy_from_memory(student_model, memory_bank, num_iterations=1000):
    """
    将记忆库中的高Q-value体验蒸馏到新模型中
    """
    for _ in range(num_iterations):
        # 从记忆库采样高Q-value体验
        high_q_exps = memory_bank.sample_by_q_threshold(threshold=0.5)
        
        # 构建蒸馏数据集
        for exp in high_q_exps:
            intent = exp.intent
            trajectory = exp.trajectory
            target_actions = [step['action'] for step in trajectory]
            
            # 监督学习：让student预测这些动作
            student_logits = student_model(intent)
            student_loss = cross_entropy(student_logits, target_actions)
            student_loss.backward()
        
        optimizer.step()
    
    return student_model
```

---

### 7.3 多游戏通用记忆库

```python
class UniversalMemoryBank:
    """跨游戏的通用记忆库"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.game_specific_banks = {}  # {game_type: SocialMemoryBank}
    
    def get_or_create_bank(self, game_type: str, player_id: int):
        """为不同游戏类型返回独立的记忆库"""
        key = f"{game_type}_{player_id}"
        if key not in self.game_specific_banks:
            # 为该游戏类型创建独立表
            db_path = f"{self.db_path}/{game_type}_p{player_id}.sqlite3"
            self.game_specific_banks[key] = SocialMemoryBank(
                db_path=db_path,
                player_id=player_id,
                game_type=game_type
            )
        return self.game_specific_banks[key]
```

---

## 总结：改造的核心价值

| 改造前（MARSHAL） | 改造后（memRL） | 获得 |
|------------------|---------------|------|
| LLM参数更新 | 记忆库Q-value更新 | 🚀 **可扩展性** |
| 模型内知识 | 显式记忆库 | 📖 **可解释性** |
| 单体Agent | 角色隔离学习 | 👥 **多智能体协作** |
| 每轮从0开始 | 检索历史经验 | 💡 **样本高效性** |
| 反向传播 | 前向推理+查表 | ⚡ **推理效率** |

---

**文档完成**。如有问题或需要进一步说明，欢迎反馈！🎯
