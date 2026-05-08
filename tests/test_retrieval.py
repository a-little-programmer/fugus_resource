from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from search_cli import build_command, eval_command, load_entities
from src.alias_matcher import build_alias_dict, match_alias
from src.embedder import CharNgramEmbedder
from src.entity_index import EntityIndex
from src.latin_expander import build_latin_lookup, expand_latin_abbreviation
from src.normalization import normalize_text


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "species_entities.jsonl"


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entities = load_entities(DATA_PATH)
        cls.entities_by_id = {entity["entity_id"]: entity for entity in cls.entities}

    def test_latin_abbreviation_format_normalization(self) -> None:
        expected = "e. coli"
        self.assertEqual(normalize_text("E. coli"), expected)
        self.assertEqual(normalize_text("E.coli"), expected)
        self.assertEqual(normalize_text("E . coli"), expected)
        self.assertEqual(normalize_text("Ｅ． ｃｏｌｉ"), expected)

    def test_exact_alias_match(self) -> None:
        alias_dict = build_alias_dict(self.entities)
        hit = match_alias(alias_dict, "金葡菌")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.entity_ids, ["TAXON:0001"])

        hit = match_alias(alias_dict, "大肠杆菌")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.entity_ids, ["TAXON:0004"])

    def test_catalog_backed_latin_expansion(self) -> None:
        lookup = build_latin_lookup(self.entities)
        expansion = expand_latin_abbreviation("S. pneumoniae", self.entities_by_id, lookup)
        self.assertIsNotNone(expansion)
        self.assertEqual(expansion.status, "expanded_exact")
        self.assertEqual(expansion.candidates[0]["scientific_name"], "Streptococcus pneumoniae")

    def test_ambiguous_abbreviation_stops_before_embedding(self) -> None:
        entities = [
            {
                "entity_id": "A",
                "standard_name_cn": "测试甲菌",
                "scientific_name": "Staphylococcus testii",
                "taxon_rank": "species",
                "aliases": [],
                "former_names": [],
            },
            {
                "entity_id": "B",
                "standard_name_cn": "测试乙菌",
                "scientific_name": "Streptococcus testii",
                "taxon_rank": "species",
                "aliases": [],
                "former_names": [],
            },
        ]
        lookup = build_latin_lookup(entities)
        entities_by_id = {entity["entity_id"]: entity for entity in entities}
        expansion = expand_latin_abbreviation("S. testii", entities_by_id, lookup)
        self.assertIsNotNone(expansion)
        self.assertEqual(expansion.status, "ambiguous_abbreviation")
        self.assertEqual(len(expansion.candidates), 2)

    def test_embedding_fallback_top3(self) -> None:
        embedder = CharNgramEmbedder()
        index = EntityIndex.build(self.entities, embedder, "char-ngram")

        results = index.search("黄色葡萄球菌", embedder, top_k=3, threshold=0.1)
        self.assertEqual(results[0].source, "embedding")
        self.assertIn("TAXON:0001", [result.entity["entity_id"] for result in results])

        results = index.search("Escherichia colli", embedder, top_k=3, threshold=0.1)
        self.assertIn("TAXON:0004", [result.entity["entity_id"] for result in results])

    def test_build_writes_alias_and_index_from_same_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            alias_path = tmp / "alias_dict.json"
            index_path = tmp / "species_index.pkl"
            with redirect_stdout(StringIO()):
                build_command(
                    Namespace(
                        data=str(DATA_PATH),
                        alias=str(alias_path),
                        index=str(index_path),
                        model="char-ngram",
                        backend="char-ngram",
                    )
                )
            self.assertTrue(alias_path.exists())
            self.assertTrue(index_path.exists())

    def test_eval_command_runs_for_threshold_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            index_path = tmp / "species_index.pkl"
            embedder = CharNgramEmbedder()
            EntityIndex.build(self.entities, embedder, "char-ngram").save(index_path)
            with redirect_stdout(StringIO()):
                eval_command(
                    Namespace(
                        index=str(index_path),
                        model="char-ngram",
                        backend="char-ngram",
                        query=["黄色葡萄球菌", "表皮葡萄球菌"],
                        top_k=2,
                        threshold=0.82,
                        format="json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
