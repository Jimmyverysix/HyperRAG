from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from research.path_consistent_negative_learning.figures.gen_retriever_only import (
    generate_main_figure,
    generate_method_figure,
    generate_prevalence_figure,
)


class RetrieverFigureTests(unittest.TestCase):
    def test_all_formal_figures_render_as_pdf_and_svg(self) -> None:
        path = [
            {"node_id": "s", "kind": "entity", "text": "Topic"},
            {"node_id": "H:1", "kind": "hyperedge", "text": "Fact one"},
            {"node_id": "a", "kind": "entity", "text": "Answer"},
        ]
        case = {
            "question": "Which answer is connected to the topic?",
            "selected_paths": [path],
            "selected_path_for_witness_answer": path,
            "conflict_witness_path": path,
            "sampled_negative_transition": {
                "head": "s",
                "hyperedge": "H:1",
                "tail": "a",
            },
            "distance_equality": {
                "topic_to_head": 0,
                "transition_cost": 2,
                "tail_to_answer": 0,
                "topic_to_answer": 2,
            },
        }
        prevalence = {
            "domains": [
                {
                    "domain": "toy",
                    "path_consistent_negative_rate": 0.05,
                    "affected_query_rate_across_seeds": 0.25,
                }
            ]
        }
        metrics = {"reciprocal_rank": 0.2}
        main = {
            "domains": [
                {
                    "domain": "toy",
                    "baseline": metrics,
                    "matched_random": {"reciprocal_rank": 0.21},
                    "ours": {"reciprocal_rank": 0.23},
                }
            ],
            "domain_comparisons": [
                {
                    "domain": "toy",
                    "reference": reference,
                    "difference_percentage_points": value,
                    "ci_low_percentage_points": value - 0.01,
                    "ci_high_percentage_points": value + 0.01,
                }
                for reference, value in (
                    ("baseline", 0.03),
                    ("matched_random", 0.02),
                )
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generate_method_figure(case, output)
            generate_prevalence_figure(prevalence, output)
            generate_main_figure(main, output)
            for stem in (
                "supervision_conflict",
                "conflict_prevalence",
                "main_effects",
            ):
                self.assertGreater((output / f"{stem}.pdf").stat().st_size, 0)
                self.assertGreater((output / f"{stem}.svg").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
