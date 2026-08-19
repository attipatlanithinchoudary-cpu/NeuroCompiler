from __future__ import annotations

import csv
import json

from scripts.reward_reliability_filter import (
    benchmark_weights,
    evaluate_summary,
    filter_csv,
    reliable_benchmarks,
    write_weighted_csv,
)


def _protocol(rank, sign, mean_abs, cvs=(1.0, 2.0, 3.0)):
    return {
        "o0_median_sec": 1.0,
        "repeatability": {
            "spearman_batch1_vs_batch2": rank,
            "sign_agreement": sign,
            "mean_abs_improvement_pct": mean_abs,
        },
        "passes": {
            f"-p{i}": {"cv_pct_mean": cv}
            for i, cv in enumerate(cvs)
        },
    }


def _summary():
    return {
        "benchmarks": {
            "benchmark://cbench-v1/gsm": {
                "suite": "cbench-v1",
                "protocols": {"B": _protocol(-0.12, 0.50, 0.50)},
            },
            "benchmark://cbench-v1/tiff2rgba": {
                "suite": "cbench-v1",
                "protocols": {"B": _protocol(0.71, 0.75, 1.72)},
            },
            "benchmark://cbench-v1/stringsearch": {
                "suite": "cbench-v1",
                "protocols": {"B": _protocol(0.24, 0.50, 5.48, cvs=(13.0, 14.0))},
            },
        }
    }


def test_reliability_gate_keeps_only_stable_reward_rankings():
    rows = evaluate_summary(
        _summary(),
        protocol="B",
        min_rank_corr=0.5,
        min_sign_agreement=0.7,
        min_mean_abs_improvement_pct=1.0,
        max_median_cv_pct=5.0,
    )
    assert reliable_benchmarks(rows) == {"benchmark://cbench-v1/tiff2rgba"}

    failures = {row["benchmark"]: row["failures"] for row in rows}
    assert "rank_corr" in failures["gsm"]
    assert "mean_abs_improvement" in failures["gsm"]
    assert "median_cv" in failures["stringsearch"]


def test_filter_csv_writes_only_reliable_benchmarks(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "filtered.csv"
    rows = [
        {"benchmark_uri": "benchmark://cbench-v1/gsm", "pass_flag": "-licm"},
        {"benchmark_uri": "benchmark://cbench-v1/tiff2rgba", "pass_flag": "-licm"},
    ]
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark_uri", "pass_flag"])
        writer.writeheader()
        writer.writerows(rows)

    kept, dropped = filter_csv(
        input_path,
        output_path,
        {"benchmark://cbench-v1/tiff2rgba"},
    )
    assert (kept, dropped) == (1, 1)

    with output_path.open(newline="", encoding="utf-8") as handle:
        out = list(csv.DictReader(handle))
    assert out == [{"benchmark_uri": "benchmark://cbench-v1/tiff2rgba", "pass_flag": "-licm"}]


def test_weighted_csv_keeps_rows_and_adds_reliability_weight(tmp_path):
    report_rows = evaluate_summary(
        _summary(),
        protocol="B",
        min_rank_corr=0.5,
        min_sign_agreement=0.7,
        min_mean_abs_improvement_pct=1.0,
        max_median_cv_pct=5.0,
    )
    weights = benchmark_weights(report_rows)

    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "weighted.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark_uri", "pass_flag"])
        writer.writeheader()
        writer.writerows(
            [
                {"benchmark_uri": "benchmark://cbench-v1/gsm", "pass_flag": "-licm"},
                {"benchmark_uri": "benchmark://cbench-v1/tiff2rgba", "pass_flag": "-licm"},
            ]
        )

    total, zero_weighted = write_weighted_csv(input_path, output_path, weights)
    assert (total, zero_weighted) == (2, 1)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["reward_reliability_weight"] == "0.000000"
    assert rows[1]["reward_reliability_weight"] == "1.000000"
