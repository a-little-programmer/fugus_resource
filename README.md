# 菌种资源库实体召回

面向菌种资源库的标准实体召回工具。系统支持将用户输入的非标准名称、简称、旧称或拉丁名缩写，匹配到库内标准菌种实体。

示例：

```text
金葡菌        -> 金黄色葡萄球菌
E.coli        -> 大肠埃希氏菌
P. fluorescens -> 荧光假单胞菌
黄色葡萄球菌  -> 金黄色葡萄球菌
```

## 功能

- 支持中文标准名、别名、曾用名的精确匹配
- 支持拉丁名缩写和格式归一，如 `E.coli`、`E . coli`、`Ｅ． ｃｏｌｉ`
- 支持向量召回，用于处理模糊表达和轻微拼写错误
- 支持低置信度标记，避免将泛查询强行归一到某个实体
- 提供命令行工具和本地 Web 页面

## 项目结构

```text
data/                  实体数据
src/                   检索、规范化、索引构建逻辑
web/                   前端页面
tests/                 单元测试
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

默认模型为 `BAAI/bge-m3`。首次运行时会从 Hugging Face 下载模型；如果模型已经缓存到本机，也可以离线运行。

## 构建索引

```bash
python search_cli.py build
```

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

## 说明

当前版本以 taxon 级实体标准化为主，不处理具体菌株编号、保藏编号或实验室株系编号，例如 `ATCC 25922`、`CGMCC`、`CICC 10001`、`DH5α`、`BL21`。这类编号需要结合字符串匹配或专门的菌株级索引处理。
