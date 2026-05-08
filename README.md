# 菌种资源库 Taxon 实体 Hybrid Embedding 召回原型

这个原型实现了三层召回流程：

1. 库内标准名、别名、曾用名的 normalized exact match。
2. 基于库内 `scientific_name` 的拉丁属名缩写解析。
3. embedding 兜底召回，同一实体多个名称向量取 max score。

默认优先使用 `sentence-transformers` 和 `BAAI/bge-m3`。如果当前环境没有安装依赖，程序会自动退到标准库字符 n-gram fallback，便于离线验证流程。

## Usage

```bash
python3 search_cli.py build --data data/species_entities.jsonl --index artifacts/species_index.pkl --alias artifacts/alias_dict.json
python3 search_cli.py query "金葡菌"
python3 search_cli.py query "E.coli"
python3 search_cli.py query "黄色葡萄球菌" --top-k 3
python3 search_cli.py eval --top-k 3
```

## Web UI

```bash
python3 search_server.py --host 127.0.0.1 --port 8000
```

然后打开 `http://127.0.0.1:8000`。如果模型已经缓存好但当前网络不可用，可以用离线模式启动：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 search_server.py
```

也可以用简化启动脚本：

```bash
python3 run_web.py
```

## Crawl CICC Product Lists

抓取 `mid=1` 下每个推荐列表页，输出候选 taxon：

```bash
python3 scripts/crawl_cicc_products.py \
  --start-url "https://www.china-cicc.org/cicc/product/?mid=1" \
  --output data/cicc_food_candidates.jsonl
```

把候选和当前 demo 数据去重合并：

```bash
python3 scripts/merge_taxon_jsonl.py \
  --base data/species_entities.jsonl \
  --candidates data/cicc_food_candidates.jsonl \
  --output data/species_entities.merged.jsonl
```

确认 `data/species_entities.merged.jsonl` 后，替换主数据并重建索引：

```bash
mv data/species_entities.merged.jsonl data/species_entities.jsonl
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 search_cli.py build
```

强制使用 fallback 后端：

```bash
python3 search_cli.py build --backend char-ngram
python3 search_cli.py query "Escherichia colli" --backend char-ngram
```
