from __future__ import annotations

import unittest

from evaluate_embedding import candidate_texts
from scripts.build_training_data import (
    build_eval_rows,
    build_training_rows,
    choose_alias_holdouts,
    split_entities,
)


class TrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.entities = [
            {
                "entity_id": "A",
                "standard_name_cn": "金黄色葡萄球菌",
                "scientific_name": "Staphylococcus aureus",
                "taxon_rank": "species",
                "aliases": ["金葡菌", "S. aureus"],
                "former_names": [],
            },
            {
                "entity_id": "B",
                "standard_name_cn": "表皮葡萄球菌",
                "scientific_name": "Staphylococcus epidermidis",
                "taxon_rank": "species",
                "aliases": ["表葡菌", "S. epidermidis"],
                "former_names": [],
            },
            {
                "entity_id": "C",
                "standard_name_cn": "大肠埃希氏菌",
                "scientific_name": "Escherichia coli",
                "taxon_rank": "species",
                "aliases": ["大肠杆菌", "E. coli"],
                "former_names": [],
            },
            {
                "entity_id": "D",
                "standard_name_cn": "肺炎链球菌",
                "scientific_name": "Streptococcus pneumoniae",
                "taxon_rank": "species",
                "aliases": ["肺炎球菌", "S. pneumoniae"],
                "former_names": [],
            },
        ]

    def test_training_rows_are_triplets(self) -> None:
        train_entities, _ = split_entities(self.entities, 0.25, seed=1)
        rows = build_training_rows(train_entities, alias_holdouts={}, max_pairs_per_entity=4)
        self.assertTrue(rows)
        self.assertTrue(all({"anchor", "positive", "hard_negative"} <= set(row) for row in rows))
        self.assertTrue(all(row["positive"] != row["hard_negative"] for row in rows))

    def test_alias_holdout_is_not_used_as_training_anchor(self) -> None:
        train_entities = self.entities[:2]
        holdouts = {"A": "金葡菌"}
        rows = build_training_rows(train_entities, holdouts, max_pairs_per_entity=4)
        self.assertNotIn("金葡菌", [row["anchor"] for row in rows])

    def test_eval_rows_include_alias_and_entity_splits(self) -> None:
        train_entities, entity_eval_entities = split_entities(self.entities, 0.25, seed=1)
        holdouts = choose_alias_holdouts(train_entities, ratio=1.0, seed=1)
        rows = build_eval_rows(train_entities, entity_eval_entities, holdouts, max_queries_per_entity=2)
        splits = {row["split"] for row in rows}
        self.assertIn("alias", splits)
        self.assertIn("entity", splits)

    def test_candidate_modes(self) -> None:
        entity = self.entities[0]
        canonical = candidate_texts(entity, "canonical")
        all_names = candidate_texts(entity, "all-names")
        self.assertIn("金黄色葡萄球菌", canonical)
        self.assertIn("Staphylococcus aureus", canonical)
        self.assertNotIn("金葡菌", canonical)
        self.assertIn("金葡菌", all_names)


if __name__ == "__main__":
    unittest.main()
