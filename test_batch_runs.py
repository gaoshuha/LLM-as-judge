"""Local, zero-cost tests for multi-version repeated experiment scheduling."""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import position_bias_experiment as experiment


def main() -> None:
    original_results_root = experiment.RESULTS_ROOT
    original_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment.RESULTS_ROOT = Path(temp_dir)
            sys.argv = [
                "position_bias_experiment.py",
                "--answer-versions", "2,3",
                "--rounds", "2",
                "--mock",
                "--limit", "1",
                "--fresh",
            ]
            with redirect_stdout(io.StringIO()):
                experiment.main()

            expected_runs = {"V2R1", "V2R2", "V3R1", "V3R2"}
            assert {path.name for path in Path(temp_dir).iterdir()} == expected_runs
            for run_name in expected_runs:
                run_dir = Path(temp_dir) / run_name
                expected_files = {
                    "judge_prompts.jsonl",
                    "judge_outputs.jsonl",
                    "mapped_results.csv",
                    "metrics_summary.csv",
                    "report.md",
                }
                assert expected_files <= {path.name for path in run_dir.iterdir()}
                output_lines = (run_dir / "judge_outputs.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                assert len(output_lines) == 4

            # The same plan without --fresh must resume each named run instead
            # of appending duplicate successful judgments.
            sys.argv.remove("--fresh")
            with redirect_stdout(io.StringIO()):
                experiment.main()
            for run_name in expected_runs:
                output_lines = (
                    Path(temp_dir) / run_name / "judge_outputs.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                assert len(output_lines) == 4
    finally:
        experiment.RESULTS_ROOT = original_results_root
        sys.argv = original_argv

    print("BATCH_PLAN_OK; NAMED_FOLDERS_OK; OUTPUT_ISOLATION_OK; RESUME_OK")


if __name__ == "__main__":
    main()
