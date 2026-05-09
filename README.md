# 菌种资源库实体召回

面向菌种资源库的领域 embedding 训练与实体召回项目。目标是微调一个更懂菌种名称、简称、旧称和拉丁名缩写的向量模型，使同一标准实体的不同表达在向量空间中更接近。

示例：

```text
金葡菌        -> 金黄色葡萄球菌
E.coli        -> 大肠埃希氏菌
P. fluorescens -> 荧光假单胞菌
黄色葡萄球菌  -> 金黄色葡萄球菌
```

## 功能

- 从菌种实体数据生成 `anchor / positive / hard_negative` 三元组训练样本
- 支持 alias split 和 entity split 两类评估集
- 支持基于 `sentence-transformers` 微调 embedding 模型
- 支持 base model 与微调模型的 topK 评估
- 支持中文标准名、别名、曾用名的精确匹配
- 支持拉丁名缩写和格式归一，如 `E.coli`、`E . coli`、`Ｅ． ｃｏｌｉ`
- 提供命令行检索和本地 Web 页面用于模型效果展示

## 项目结构

```text
data/                  实体数据
src/                   检索、规范化、索引构建逻辑
web/                   前端页面
tests/                 单元测试
train_embedding.py     embedding 微调入口
evaluate_embedding.py  embedding 评估入口
search_cli.py          命令行入口
search_server.py       本地 Web 服务
run_web.py             Web 页面启动脚本
requirements.txt       Python 依赖
```

## 环境准备

建议使用 Python 3.11。

```bash
conda create --prefix ./.conda-env python=3.11 -y
conda activate ./.conda-env
pip install -r requirements.txt
```

训练原型默认使用 `BAAI/bge-small-zh-v1.5`。该模型比 `BAAI/bge-m3` 更轻，适合先跑通训练和评估流程。后续可以在显存充足时切换到更大的模型。

## 生成训练数据

```bash
python scripts/build_training_data.py
```

默认输出：

```text
data/train_triplets.jsonl
data/eval_queries.jsonl
```

训练样本格式：

```json
{
  "anchor": "金葡菌",
  "positive": "金黄色葡萄球菌",
  "hard_negative": "表皮葡萄球菌",
  "entity_id": "TAXON:0001"
}
```

其中 `hard_negative` 优先来自同属不同种或名称相近实体。

## 微调模型

```bash
python train_embedding.py \
  --train data/train_triplets.jsonl \
  --output models/fugus-entity-embedding \
  --epochs 1 \
  --batch-size 16
```

如果模型已经下载到本机但当前网络不稳定，可以加 `--offline`：

```bash
python train_embedding.py --epochs 1 --batch-size 16 --offline
```

显存较小时可以降低 batch size：

```bash
python train_embedding.py \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --fp16
```

`--gradient-accumulation-steps` 依赖 `datasets` / `SentenceTransformerTrainer` 路径；如果环境中没有 `datasets`，脚本会退回到 `SentenceTransformer.fit` 训练路径，但不支持梯度累加。

如果要尝试更大的模型：

```bash
python train_embedding.py --base-model BAAI/bge-m3 --batch-size 4 --fp16
```

## 评估模型

```bash
python evaluate_embedding.py
```

默认会评估 base model；如果存在 `models/fugus-entity-embedding`，会同时评估微调模型。

评估候选有两种模式：

```bash
python evaluate_embedding.py --candidate-mode canonical
python evaluate_embedding.py --candidate-mode all-names
```

离线评估：

```bash
python evaluate_embedding.py --candidate-mode canonical --offline
```

一次运行两种评估，并分别保存报告：

```bash
python run_eval.py
```

`canonical` 只把标准中文名和拉丁名作为候选，用于观察模型是否把非标准表达拉近到标准表达。`all-names` 会把别名和曾用名也作为候选，更接近真实检索效果，但指标通常会更高。

输出指标包括：

```text
top1 accuracy
top3 recall
MRR
```

详细结果写入：

```text
reports/embedding_eval.json
```

## 当前实验结果

当前数据集包含 154 个 taxon 实体，评估集包含 136 条查询，其中 alias split 44 条、entity split 92 条。

`canonical` 模式只使用标准中文名和标准拉丁名作为候选，更适合衡量模型是否学会把非标准表达拉近到标准表达。

| 模型 | Split | Top1 Accuracy | Top3 Recall | MRR | 查询数 |
| --- | --- | ---: | ---: | ---: | ---: |
| BAAI/bge-small-zh-v1.5 | all | 0.8971 | 0.9485 | 0.9267 | 136 |
| models/fugus-entity-embedding | all | 0.9265 | 0.9412 | 0.9393 | 136 |
| BAAI/bge-small-zh-v1.5 | entity | 0.9239 | 0.9783 | 0.9542 | 92 |
| models/fugus-entity-embedding | entity | 0.9674 | 0.9783 | 0.9783 | 92 |
| BAAI/bge-small-zh-v1.5 | alias | 0.8409 | 0.8864 | 0.8693 | 44 |
| models/fugus-entity-embedding | alias | 0.8409 | 0.8636 | 0.8580 | 44 |

`all-names` 模式会把别名和曾用名也放入候选，更接近真实检索场景，但不适合单独证明模型学会了标准化。

| 模型 | Split | Top1 Accuracy | Top3 Recall | MRR | 查询数 |
| --- | --- | ---: | ---: | ---: | ---: |
| BAAI/bge-small-zh-v1.5 | all | 1.0000 | 1.0000 | 1.0000 | 136 |
| models/fugus-entity-embedding | all | 0.9632 | 1.0000 | 0.9804 | 136 |
| BAAI/bge-small-zh-v1.5 | entity | 1.0000 | 1.0000 | 1.0000 | 92 |
| models/fugus-entity-embedding | entity | 0.9565 | 1.0000 | 0.9783 | 92 |
| BAAI/bge-small-zh-v1.5 | alias | 1.0000 | 1.0000 | 1.0000 | 44 |
| models/fugus-entity-embedding | alias | 0.9773 | 1.0000 | 0.9848 | 44 |

在 `canonical` 评估中，微调模型的整体 Top1 从 0.8971 提升到 0.9265，entity split Top1 从 0.9239 提升到 0.9674，说明微调后模型对标准实体表达的对齐能力有所提升。

## 构建索引

```bash
python search_cli.py build
```

默认优先加载 `models/fugus-entity-embedding`。如果本地没有微调模型，会回退到 base model 并打印 warning。

构建完成后会生成：

```text
artifacts/alias_dict.json
artifacts/species_index.pkl
```

## 命令行查询

```bash
python search_cli.py query "金葡菌"
python search_cli.py query "E.coli"
python search_cli.py query "黄色葡萄球菌" --top-k 3
```

输出字段包括标准中文名、拉丁名、命中来源、相似度分数和置信度状态。

## 启动 Web 页面

```bash
python run_web.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果需要指定端口：

```bash
python run_web.py --host 127.0.0.1 --port 8010
```

## 数据格式

主数据文件为：

```text
data/species_entities.jsonl
```

每行一个 JSON 对象：

```json
{
  "entity_id": "TAXON:0001",
  "standard_name_cn": "金黄色葡萄球菌",
  "scientific_name": "Staphylococcus aureus",
  "taxon_rank": "species",
  "aliases": ["金葡菌", "S. aureus"],
  "former_names": ["Micrococcus aureus"],
  "metadata": {
    "genus_cn": "葡萄球菌属"
  }
}
```

`aliases` 和 `former_names` 会参与精确匹配和向量索引。`metadata` 只作为补充信息保存，不参与召回逻辑。

## 测试

```bash
python -m unittest discover -s tests
```

## 非标准表达召回报告

生成只包含 `aliases` 和 `former_names` 的召回自检报告：

```bash
python scripts/report_non_standard_recall.py
```

默认输出：

```text
reports/non_standard_recall_inputs.json
```

报告中的 `input` 是实际查询入参，`expected` 是期望标准实体，`actual_top1` 是系统返回的第一名。

## 说明

当前版本以 taxon 级实体标准化为主，不处理具体菌株编号、保藏编号或实验室株系编号，例如 `ATCC 25922`、`CGMCC`、`CICC 10001`、`DH5α`、`BL21`。这类编号需要结合字符串匹配或专门的菌株级索引处理。

Web 和 CLI 是模型效果展示入口，不是训练目标本身。项目核心目标是训练一个让菌种非标准表达与标准表达更接近的领域 embedding 模型。
