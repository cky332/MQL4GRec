# MQL4GRec 对抗鲁棒性实验:像素级 PGD 商品推广攻击(MLLM-MSR 移植版)

把在 MLLM-MSR 上奏效的攻击移植到 MQL4GRec:扰动候选商品**封面像素**(L∞≤ε,在公开 CLIP 上做
PGD),让其 CLIP 嵌入靠近**热门 top-20 商品的嵌入质心**,看该商品排名是否上升。

**与 MLLM-MSR 的本质区别**:MQL4GRec 把 CLIP 嵌入**离线量化成离散码**后,推荐模型只看码。所以攻击
只有在像素扰动**翻转了量化码**(尤其是翻向热门码前缀)时才有效。本工具把这一点**直接测量**出来
——若 16/255 跨不过量化边界导致码不变、攻击被钝化,这本身就是相对 MLLM-MSR 的鲁棒性发现。

## 前置条件
1. 已微调的推荐模型 `./log/Instruments/`(跑 `scripts/run_local.sh` 得到)。
2. **作者的 RQ-VAE 检查点**(从项目 Google Drive),放到
   `index/log/Instruments/ViT-L-14_256/best_collision_model.pth`。
3. Amazon-2018 元数据目录,含 `meta_Musical_Instruments.json.gz`(下载受害封面用)。
4. 依赖与主项目一致(`requirements.txt`:torch / torchvision / transformers≤4.45 / numpy<2 / Pillow / requests)。

## 两种攻击模式

- **`MODE=pixel`(默认,忠实 MLLM-MSR 移植)**:对封面**像素**做 PGD(L∞≤ε),需要原始图片
  → 需 `META_DATA_PATH`(Amazon 元数据)。
- **`MODE=embedding`(兜底,无需图片/元数据/CLIP)**:直接对商品**已有的 CLIP 嵌入**做 PGD
  (L2 预算 `rho·‖x‖`),只需 RQ-VAE。不那么"忠实"(略去像素→CLIP 这段),但能立刻测出
  核心问题(量化是否被翻、ensemble 是否稀释)。

## 一键运行
```bash
# 嵌入层模式(立刻可跑,不需要图片):
RQVAE_CKPT=index/log/Instruments/ViT-L-14_256/best_collision_model.pth \
MODE=embedding bash attack/run_attack.sh

# 像素模式(忠实版,需要 Amazon 元数据):
META_DATA_PATH=/path/to/amazon18/Metadata \
RQVAE_CKPT=index/log/Instruments/ViT-L-14_256/best_collision_model.pth \
bash attack/run_attack.sh
```
可调环境变量:`MODE`(pixel/embedding)、`NUM_TASKS`(默认 200)、`EPS_LIST`(像素,默认
`"16 32 64"`,/255)、`RHO_LIST`(嵌入,默认 `"10 20 30 50"`,即 %)、`STEPS`、`MODEL_CKPT`、
`GPUS`(默认 0,单卡)。

## 推广测试 vs 反噬测试(攻击对象的选择)

- **默认(standard)**:攻击对象 = 用户的**真实下一个商品**(正样本)。它本就排得高,改码只会更差
  → 测的是**鲁棒性/反噬**,不是推广。
- **`PROMOTE=1`(promote)**:攻击对象 = 用户**没交互过**的**冷门目标商品** T,在 `[T] + (真实正样本
  + 随机非交互负样本)` 中给 T 排名 → 测**能否把 T 推进用户的前列**(`rank` 越低=推广越成功)。
  目标商品默认取交互最少的 `N_TARGETS`(默认 5)个;也可 `--targets id1,id2` 指定。

```bash
# 推广测试(冷门商品,嵌入模式,立刻可跑):
RQVAE_CKPT=index/log/Instruments/ViT-L-14_256/best_collision_model.pth \
MODE=embedding PROMOTE=1 N_TARGETS=5 bash attack/run_attack.sh
```

## 流水线(也可单步运行,均在 `attack/` 目录下)
| 阶段 | 脚本 | 作用 | 产物(`attack/artifacts/`) |
|---|---|---|---|
| 1 | `sample_tasks.py` | 采样 (用户, 受害正样本, 20 负样本) 任务并固定 | `tasks.json`, `victims.json` |
| 2 | `download_subset.py` | 只下载**受害商品**封面(热门质心用 shipped 嵌入,无需下热门图) | `victim_images.json`, `images/` |
| 3 | `clip_attack.py` | 对每张受害封面做 PGD,推向热门质心 | `adv_emb_eps{E}.npz`, `clean_emb.npz`, `attack_diag_eps{E}.json` |
| 4 | `requantize.py` | 用 RQ-VAE 把扰动嵌入量化成码,替换进 shipped 索引 | `index_vitemb_ATTACKED_eps{E}.json`, `requant_diag_eps{E}.json` |
| 5 | `run_eval.py` | 1+20 评测:干净 vs 攻击,纯图像通道 vs ensemble | `eval_*.json` |

## 读结果
`run_eval.py` 打印四象限对照(纯图像/ensemble × 干净/攻击),指标:**平均排名、hit@10、NDCG@10、
P(Yes) 类比量**,并给出**码变化率 / 是否移向热门前缀**。预期看到:
- **码变化率随 ε**:16/255 可能近零(量化吸收了扰动)→ 攻击被钝化;ε 增大后码开始翻转。
- **纯图像 ≫ ensemble**:纯图像通道效果最强,而 ensemble 与未受攻击的文本通道平均后被稀释。

## 设计要点 / 注意
- **质心来自 shipped `.npy`**(已是全部商品的 CLIP 嵌入),故只需下载受害图片。
- **替换进 shipped**:干净索引=原版 shipped(模型微调所用),攻击索引仅替换受害商品行 → 仅一处差异,
  完全受控,**无需重新微调**。`requantize.py` 会校验"未扰动受害商品重量化是否==shipped"以确认一致性。
- **PGD 用 fp32**(CUDA 上 CLIP 默认 fp16 会让梯度不稳);用的是仓库自带的 `data_process/clip`,与离线
  特征提取逐字节一致 → 迁移无损。
- **P(Yes) 类比量**:对 21 个候选打分做 softmax 后落在受害商品上的概率(生成式模型里最接近 MLLM-MSR
  的 P(Yes))。注意:在"21 候选都在两通道打分"的逐点设定下,`ensemble.py` 的 `+1` 一致性奖励对所有候选
  恒定、对排名无影响,故 ensemble 退化为两通道分数的平均(这正是稀释的来源)。
- **图片漂移**:`clean_emb` 与 shipped 嵌入若余弦 <0.99(2018 年后图片可能变更),干净基线会不准——
  `requantize.py` 的 `clean_regen_matches_shipped` 即用于发现此情况。
- 本目录所有脚本用普通 `python`(单进程),不走 `torchrun`,故无 DDP/NCCL 与模型并行问题。
