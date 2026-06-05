# 本地部署 / 跑通指南 (Linux + Anaconda)

本文档说明如何在本地单卡(或多卡)环境跑通 MQL4GRec **Instruments** 数据集的
完整推荐流程:**微调 → 推理(文本通道 + 图像通道)→ 融合(ensemble)**。

仓库已自带 `data/Instruments/` 的**预生成语义ID编码**(`*.index_lemb.json` 文本码、
`*.index_vitemb.json` 图像码),因此**无需**下载商品图片、跑 CLIP、训练 RQ-VAE 量化器
——离线预处理这部分最重的步骤可以全部跳过。

> 仅当你想从原始图片重新生成编码时,才需要 `data_process/` 与 `index/` 流程
> (见本文末尾"可选:从零重建编码")。

---

## 1. 创建 conda 环境

```bash
conda create -n mql4grec python=3.10 -y
conda activate mql4grec

# 1) 先单独装 PyTorch,按你的 CUDA 版本选 wheel(用 nvidia-smi 看驱动/CUDA)
#    CUDA 12.1:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
#    或 CUDA 11.8:
# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 2) 再装其余依赖
pip install -r requirements.txt
```

**版本注意**:`transformers` 必须 `<= 4.45.0`(代码用了旧的 `evaluation_strategy`
参数名,4.46 起被移除;README 也实测 4.45.0 收敛良好)。`numpy` 必须 `< 2.0`。

验证安装:
```bash
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__)"
```

---

## 2. 一键跑通(推荐)

```bash
# 忠实设置(epochs=200,带早停,会自动在验证集不再提升时停止)
bash scripts/run_local.sh

# 想先快速验证"能跑通"(几分钟级,指标会很差,只为验证管线):
EPOCHS=2 BATCH=128 bash scripts/run_local.sh
```

常用环境变量覆盖:`EPOCHS`、`BATCH`、`NUM_BEAMS`、`GPUS`(=nproc_per_node)、
`DATASET`、`LOAD_MODEL_NAME`(从预训练 ckpt 续训)。

多卡示例:
```bash
GPUS=2 bash scripts/run_local.sh
```

跑完后结果在 `./log/Instruments/`:
- `results_seqrec_20.json`   —— 纯文本通道指标
- `results_seqimage_20.json` —— 纯图像通道指标
- `results_ensemble_20.json` —— **图文融合后的最终指标**(论文汇报口径)

指标含义:`hit@1,hit@5,hit@10,ndcg@5,ndcg@10`。

---

## 3. 想复现论文级别效果?(可选:加载预训练权重)

`scripts/run_local.sh` 默认**从头**在 Instruments 上训练那个小 T5(跳过了跨域预训练),
能完整跑通并出真实指标,但会低于论文值。论文流程是"多域预训练 → 单域微调"。

如果你想要论文级效果,从项目 Google Drive 下载预训练 ckpt:
<https://drive.google.com/drive/folders/1eewycbcAJ95atmF_V3bNchPIFDSw_TQC>
然后:
```bash
LOAD_MODEL_NAME=/path/to/downloaded/pretrain_ckpt bash scripts/run_local.sh
```

(本地自己跑预训练 `scripts/pretrain.sh` 需要 Pet/Cell/Automotive/Tools/Toys/Sports
等多个数据集,仓库只带了 Instruments,故本地无法直接复跑预训练。)

---

## 4. 分步手动运行(等价于一键脚本,便于调试)

```bash
DATASET=Instruments
OUT=./log/$DATASET
mkdir -p $OUT

# (1) 微调
torchrun --nproc_per_node=1 --master_port=2309 finetune.py \
  --base_model ./config/ckpt --data_path ./data/ --dataset $DATASET \
  --output_dir $OUT --per_device_batch_size 256 --learning_rate 5e-4 \
  --epochs 200 --weight_decay 0.01 --save_and_eval_strategy epoch \
  --logging_step 50 --max_his_len 20 --prompt_num 4 --patient 10 \
  --index_file .index_lemb.json --image_index_file .index_vitemb.json \
  --tasks seqrec,seqimage,item2image,image2item,fusionseqrec --valid_task seqrec

# (2) 推理:两个通道都要跑(ensemble 需要两份 save 文件)
for TASK in seqrec seqimage; do
  torchrun --nproc_per_node=1 --master_port=2309 test_ddp_save.py \
    --ckpt_path $OUT --data_path ./data/ --dataset $DATASET \
    --test_batch_size 64 --num_beams 20 \
    --index_file .index_lemb.json --image_index_file .index_vitemb.json \
    --test_task $TASK \
    --results_file $OUT/results_${TASK}_20.json \
    --save_file $OUT/save_${TASK}_20.json --filter_items
done

# (3) 融合
python ensemble.py --output_dir $OUT --dataset $DATASET --data_path ./data/ \
  --index_file .index_lemb.json --image_index_file .index_vitemb.json --num_beams 20
```

---

## 5. 常见问题排查

- **`NotImplementedError: Using RTX 4000 series doesn't support ... P2P or IB`**
  (RTX 40 系消费卡,如 4090):`run_local.sh` 已内置
  `export NCCL_P2P_DISABLE=1` 与 `export NCCL_IB_DISABLE=1` 修复。若你手动用
  `torchrun` 跑,需自己先 export 这两个变量。
- **`ImportError: cannot import name 'ItemImageDelDataset'`**:已修复(`utils.py` 原本
  import 了 5 个未随仓库提供的数据集类)。如仍出现,确认你在本分支上。注意:这意味着
  `*del` / `*fg*` 这些任务本地不可用(对应类没随仓库发布),但文档中的 5 个任务不受影响。
- **CUDA out of memory**:调小 `BATCH`(如 `BATCH=64`)和/或推理 `--test_batch_size`;
  也可给 finetune 加 `--fp16`。
- **单卡但报 DDP / NCCL 相关错误**:`test_ddp_save.py` 走分布式,单卡用
  `--nproc_per_node=1` 即可(脚本已默认)。
- **`evaluation_strategy` 报错**:说明 transformers 版本过新,降到 `4.45.0`。
- **只有 CPU**:可勉强跑通(模型很小),但 200 epochs 不现实,建议 `EPOCHS=2` 验证管线;
  代码默认走 CUDA,纯 CPU 可能需要少量改动。

---

## 6. 可选:从零重建编码(从原始图片)

仅当你不想用自带编码、要自己复现整条离线链路时:
```bash
cd data_process
bash 1_load_figure.sh   # 下载图片(每个商品取一张代表图)
bash 2_process.sh       # 清洗:保证每个 item 一图一文本
bash 3_get_text_emb.sh  # LLaMA 文本 embedding  -> .emb-llama-td.npy
bash 4_get_image_emb.sh # CLIP ViT-L/14 图像 embedding -> .emb-ViT-L-14.npy
cd ../index
bash scripts/run.sh         # 训练 RQ-VAE 量化器
bash scripts/gen_code_dis.sh # 生成 .index_lemb.json / .index_vitemb.json
```
这部分需要额外依赖(见 `requirements.txt` 末尾被注释的可选项)与 Amazon 原始数据,
且耗时较长。
