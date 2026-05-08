#!/usr/bin/env python3
"""Crawl public CICC product/recommendation pages into taxon JSONL candidates.

This script intentionally extracts taxon-level names only. Strain/catalog
numbers remain metadata and are not written as aliases.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_URL = "https://www.china-cicc.org"
DEFAULT_START_URL = "https://www.china-cicc.org/cicc/product/?mid=3"
LATIN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[a-z][a-z.-]+)+(?:\s+subsp\.\s+[a-z-]+)?)\b")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9（）().·\- ]{1,40}")
DETAIL_RE = re.compile(r"/cicc/detail/\?cid=\d+")
CN_LABEL = r"(?:菌株中文名|菌种中文名|中文名称|中文名|名称)"
LATIN_LABEL = r"(?:菌株拉丁名|菌种拉丁名|拉丁名称|拉丁名|学名)"


@dataclass(frozen=True)
class Link:
    href: str
    text: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[Link] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._href = href
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            text = clean_text("".join(self._parts))
            self.links.append(Link(self._href, text))
            self._href = None
            self._parts = []


def fetch(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "taxon-demo-crawler/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        return raw.decode("utf-8", errors="replace")
    except Exception:
        if "://www.china-cicc.org" in url:
            fallback = url.replace("://www.china-cicc.org", "://china-cicc.org", 1)
            request = Request(fallback, headers={"User-Agent": "taxon-demo-crawler/0.1"})
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace")
        raise


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def page_url(base_url: str, pageid: int) -> str:
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    if pageid > 0:
        query["pageid"] = [str(pageid)]
    else:
        query.pop("pageid", None)
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def parse_links(document: str, url: str) -> list[Link]:
    parser = LinkParser()
    parser.feed(document)
    return [Link(urljoin(url, link.href), link.text) for link in parser.links]


def discover_category_urls(start_url: str, timeout: int) -> list[str]:
    document = fetch(start_url, timeout)
    urls = {start_url}
    for link in parse_links(document, start_url):
        parsed = urlparse(link.href)
        if parsed.path.endswith("/cicc/product/") and "mid=3" in parsed.query:
            urls.add(link.href)
    return sorted(urls)


def discover_recommendation_urls(start_url: str, timeout: int) -> list[tuple[str, str]]:
    document = fetch(start_url, timeout)
    urls: dict[str, str] = {}
    for link in parse_links(document, start_url):
        parsed = urlparse(link.href)
        path_query = parsed.path + ("?" + parsed.query if parsed.query else "")
        if DETAIL_RE.search(path_query):
            label = link.text.replace("查看详情", "").strip() or "CICC推荐列表"
            urls[link.href] = label
    return sorted((url, label) for url, label in urls.items())


def extract_candidates(document: str, page: str, category: str | None = None) -> list[dict]:
    candidates: list[dict] = []
    rows = split_candidate_rows(document)
    for row in rows:
        text = clean_text(re.sub(r"<[^>]+>", " ", row))
        structured = extract_structured_name(text)
        if structured:
            chinese, latin = structured
            cid = extract_cid(row)
            candidates.append(make_candidate(chinese, latin, page, category, cid))
            continue

        latin_match = LATIN_RE.search(text)
        if not latin_match:
            continue
        latin = latin_match.group(1)
        chinese = extract_chinese_name(text, latin)
        if not chinese:
            continue
        cid = extract_cid(row)
        candidates.append(make_candidate(chinese, latin, page, category, cid))
    if candidates:
        return candidates
    return extract_columnar_candidates(document, page, category)


def extract_columnar_candidates(document: str, page: str, category: str | None = None) -> list[dict]:
    items = extract_list_items(document)
    if not items:
        return []

    try:
        header_indexes = [
            items.index("序号"),
            items.index("菌株编号"),
            items.index("菌株中文名"),
            items.index("菌株拉丁名"),
        ]
    except ValueError:
        return []

    start = max(header_indexes) + 1
    while start < len(items) and items[start] in {"具体用途", "价格", "操作"}:
        start += 1

    serial_count = 0
    expected = 1
    while start + serial_count < len(items):
        value = items[start + serial_count]
        if value != str(expected):
            break
        serial_count += 1
        expected += 1

    if serial_count == 0:
        return []

    strain_start = start + serial_count
    chinese_start = strain_start + serial_count
    latin_start = chinese_start + serial_count
    if latin_start + serial_count > len(items):
        return []

    strain_numbers = items[strain_start:chinese_start]
    chinese_names = items[chinese_start:latin_start]
    latin_names = items[latin_start:latin_start + serial_count]

    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for strain_no, chinese, latin in zip(strain_numbers, chinese_names, latin_names):
        chinese = clean_chinese_name(chinese)
        latin = clean_latin_name(latin)
        if not chinese or should_skip_chinese(chinese):
            continue
        if not LATIN_RE.fullmatch(latin):
            continue
        key = (chinese, latin.casefold())
        if key in seen:
            continue
        seen.add(key)
        candidate = make_candidate(chinese, latin, page, category, None)
        candidate["metadata"]["strain_examples"] = [strain_no]
        candidates.append(candidate)
    return candidates


def extract_list_items(document: str) -> list[str]:
    raw_items = re.findall(r"<li\b[^>]*>(.*?)</li>", document, flags=re.I | re.S)
    if not raw_items:
        raw_items = re.findall(r"<td\b[^>]*>(.*?)</td>", document, flags=re.I | re.S)
    items = []
    for item in raw_items:
        text = clean_text(re.sub(r"<[^>]+>", " ", item))
        if text:
            items.append(text)
    return items


def split_candidate_rows(document: str) -> list[str]:
    rows = re.findall(r"<tr\b.*?</tr>", document, flags=re.I | re.S)
    if rows:
        return rows
    rows = re.findall(r"<li\b.*?</li>", document, flags=re.I | re.S)
    if rows:
        return rows
    return re.split(r"</p>|</div>|<br\s*/?>", document, flags=re.I)


def extract_structured_name(text: str) -> tuple[str, str] | None:
    latin_pattern = LATIN_RE.pattern
    chinese_pattern = r"([\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9（）().·\- ]{1,40}?)"
    patterns = [
        re.compile(CN_LABEL + r"\s*[:：]?\s*" + chinese_pattern + r"\s+" + LATIN_LABEL + r"\s*[:：]?\s*" + latin_pattern),
        re.compile(LATIN_LABEL + r"\s*[:：]?\s*" + latin_pattern + r"\s+" + CN_LABEL + r"\s*[:：]?\s*" + chinese_pattern),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        groups = [group for group in match.groups() if group]
        chinese = next((group for group in groups if re.search(r"[\u4e00-\u9fff]", group)), None)
        latin = next((group for group in groups if LATIN_RE.fullmatch(group)), None)
        if chinese and latin:
            chinese = clean_chinese_name(chinese)
            if chinese and not should_skip_chinese(chinese):
                return chinese, latin
    return None


def extract_chinese_name(text: str, latin: str) -> str | None:
    before_latin = text.split(latin, 1)[0]
    matches = [match.group(0).strip(" -:：|") for match in CHINESE_RE.finditer(before_latin)]
    matches = [clean_chinese_name(match) for match in matches if not should_skip_chinese(match)]
    matches = [match for match in matches if match]
    if matches:
        return matches[-1]
    return None


def clean_chinese_name(value: str) -> str:
    value = re.sub(r"^(菌株中文名|菌种中文名|中文名称|中文名|名称)\s*[:：]?", "", value)
    value = re.sub(r"(菌株拉丁名|菌种拉丁名|拉丁名称|拉丁名|学名).*$", "", value)
    return value.strip(" -:：|")


def clean_latin_name(value: str) -> str:
    value = re.sub(r"^(菌株拉丁名|菌种拉丁名|拉丁名称|拉丁名|学名)\s*[:：]?", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -:：|")


def should_skip_chinese(value: str) -> bool:
    skip_words = ("产品", "目录", "菌种", "查看", "详情", "用途", "编号", "推荐", "分类")
    return any(word in value for word in skip_words)


def extract_cid(row: str) -> str | None:
    match = re.search(r"/cicc/detail/\?cid=(\d+)", row)
    return match.group(1) if match else None


def latin_aliases(latin: str) -> list[str]:
    parts = latin.split()
    aliases = []
    if len(parts) >= 2:
        aliases.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
    return aliases


def make_candidate(
    chinese: str,
    latin: str,
    page: str,
    category: str | None,
    cid: str | None,
) -> dict:
    return {
        "standard_name_cn": chinese,
        "scientific_name": latin,
        "taxon_rank": "subspecies" if " subsp. " in latin else "species",
        "aliases": latin_aliases(latin),
        "former_names": [],
        "metadata": {
            "source": "CICC",
            "source_page": page,
            "category": category,
            "detail_url": f"{BASE_URL}/cicc/detail/?cid={cid}" if cid else None,
        },
    }


def crawl(start_url: str, max_pages: int, delay: float, timeout: int) -> list[dict]:
    recommendation_urls = discover_recommendation_urls(start_url, timeout)
    if recommendation_urls:
        return crawl_recommendations(recommendation_urls, delay, timeout)

    category_urls = discover_category_urls(start_url, timeout)
    by_latin: dict[str, dict] = {}
    for category_url in category_urls:
        empty_pages = 0
        for pageid in range(max_pages):
            url = page_url(category_url, pageid)
            document = fetch(url, timeout)
            candidates = extract_candidates(document, url)
            if not candidates:
                empty_pages += 1
                if empty_pages >= 2:
                    break
            else:
                empty_pages = 0
            for candidate in candidates:
                by_latin.setdefault(candidate["scientific_name"].casefold(), candidate)
            time.sleep(delay)
    return list(by_latin.values())


def debug_page(start_url: str, timeout: int) -> None:
    document = fetch(start_url, timeout)
    links = parse_links(document, start_url)
    detail_links = []
    for link in links:
        parsed = urlparse(link.href)
        path_query = parsed.path + ("?" + parsed.query if parsed.query else "")
        if DETAIL_RE.search(path_query):
            detail_links.append(link)

    text = clean_text(re.sub(r"<[^>]+>", " ", document))
    latin_matches = LATIN_RE.findall(text)
    rows = split_candidate_rows(document)
    print(f"URL: {start_url}")
    print(f"HTML bytes: {len(document.encode('utf-8'))}")
    print(f"Links: {len(links)}")
    print(f"Detail links: {len(detail_links)}")
    for index, link in enumerate(detail_links[:20], start=1):
        print(f"  detail[{index}]: {link.text} -> {link.href}")
    print(f"Latin-like matches in full text: {len(latin_matches)}")
    print(f"Candidate rows: {len(rows)}")
    print("Text preview:")
    print(text[:2000])


def crawl_recommendations(
    recommendation_urls: list[tuple[str, str]],
    delay: float,
    timeout: int,
) -> list[dict]:
    by_latin: dict[str, dict] = {}
    for url, label in recommendation_urls:
        document = fetch(url, timeout)
        for candidate in extract_candidates(document, url, label):
            current = by_latin.get(candidate["scientific_name"].casefold())
            if current is None:
                by_latin[candidate["scientific_name"].casefold()] = candidate
            else:
                merge_source_example(current, candidate)
        time.sleep(delay)
    return list(by_latin.values())


def merge_source_example(target: dict, candidate: dict) -> None:
    source_page = candidate.get("metadata", {}).get("source_page")
    category = candidate.get("metadata", {}).get("category")
    examples = target.setdefault("metadata", {}).setdefault("source_examples", [])
    if source_page and not any(example.get("url") == source_page for example in examples):
        examples.append({"source": "CICC", "category": category, "url": source_page})


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl public CICC product taxon candidates")
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--output", default="data/cicc_product_candidates.jsonl")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        debug_page(args.start_url, args.timeout)
        return

    candidates = crawl(args.start_url, args.max_pages, args.delay, args.timeout)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for index, candidate in enumerate(candidates, start=1):
            candidate = dict(candidate)
            candidate["entity_id"] = f"CICC_TAXON:{index:05d}"
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
