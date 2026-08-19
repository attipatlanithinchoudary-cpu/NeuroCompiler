#!/usr/bin/env python3
"""Filter reward datasets using reward-fidelity pilot repeatability.

The runtime reward diagnosis showed that not every benchmark provides a stable
ordering of LLVM pass effects. This tool turns
``results/rf_pilot_summary.json`` into a concrete data-quality gate:

* summarize which benchmark/protocol pairs pass repeatability thresholds;
* optionally filter an SL/RL CSV so unreliable benchmark rewards are excluded;
* optionally annotate a CSV with ``reward_reliability_weight`` so noisy
  rewards are downweighted instead of deleted.

By default the filter evaluates protocol B, the high-fidelity protocol used by
the O3 runtime harness. Protocol A can be selected for training-time protocol
audits.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _median_cv(protocol_summary: Dict[str, Any]) -> Optional[float]:
    values: List[float] = []
    for pass_summary in protocol_summary.get("passes", {}).values():
        value = _finite_float(pass_summary.get("cv_pct_mean"))
        if value is not None:
            values.append(value)
    return statistics.median(values) if values else None


def _benchmark_name(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def reliability_weight(
    *,
    reliable: bool,
    failures: Sequence[str],
    unreliable_weight: float,
    partial_weight: float,
) -> float:
    """Map a threshold report to a sample weight.

    Fully reliable benchmarks receive weight 1.0. Benchmarks that fail only
    the effect-size threshold are still repeatable but weak, so they receive a
    partial weight. Benchmarks with rank/sign/CV failures are too noisy to use
    as strong labels and receive ``unreliable_weight``.
    """
    if reliable:
        return 1.0
    failure_set = set(failures)
    if failure_set and failure_set <= {"mean_abs_improvement"}:
        return partial_weight
    return unreliable_weight


def evaluate_protocol(
    benchmark_uri: str,
    protocol_name: str,
    protocol_summary: Dict[str, Any],
    *,
    min_rank_corr: float,
    min_sign_agreement: float,
    min_mean_abs_improvement_pct: float,
    max_median_cv_pct: float,
    unreliable_weight: float = 0.0,
    partial_weight: float = 0.5,
) -> Dict[str, Any]:
    repeatability = protocol_summary.get("repeatability") or {}
    rank_corr = _finite_float(repeatability.get("spearman_batch1_vs_batch2"))
    sign_agreement = _finite_float(repeatability.get("sign_agreement"))
    mean_abs = _finite_float(repeatability.get("mean_abs_improvement_pct"))
    median_cv = _median_cv(protocol_summary)

    failures: List[str] = []
    if rank_corr is None or rank_corr < min_rank_corr:
        failures.append("rank_corr")
    if sign_agreement is None or sign_agreement < min_sign_agreement:
        failures.append("sign_agreement")
    if mean_abs is None or mean_abs < min_mean_abs_improvement_pct:
        failures.append("mean_abs_improvement")
    if median_cv is None or median_cv > max_median_cv_pct:
        failures.append("median_cv")

    reliable = not failures
    weight = reliability_weight(
        reliable=reliable,
        failures=failures,
        unreliable_weight=unreliable_weight,
        partial_weight=partial_weight,
    )

    return {
        "benchmark_uri": benchmark_uri,
        "benchmark": _benchmark_name(benchmark_uri),
        "suite": benchmark_uri.split("://", 1)[1].split("/", 1)[0] if "://" in benchmark_uri else "",
        "protocol": protocol_name,
        "reliable": reliable,
        "weight": weight,
        "failures": failures,
        "o0_median_sec": protocol_summary.get("o0_median_sec"),
        "median_cv_pct": median_cv,
        "sign_agreement": sign_agreement,
        "rank_corr": rank_corr,
        "mean_abs_improvement_pct": mean_abs,
    }


def evaluate_summary(
    summary: Dict[str, Any],
    *,
    protocol: str,
    min_rank_corr: float,
    min_sign_agreement: float,
    min_mean_abs_improvement_pct: float,
    max_median_cv_pct: float,
    unreliable_weight: float = 0.0,
    partial_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for benchmark_uri, benchmark_summary in sorted(summary.get("benchmarks", {}).items()):
        protocols = benchmark_summary.get("protocols", {})
        protocol_summary = protocols.get(protocol)
        if protocol_summary is None:
            rows.append(
                {
                    "benchmark_uri": benchmark_uri,
                    "benchmark": _benchmark_name(benchmark_uri),
                    "suite": benchmark_summary.get("suite", ""),
                    "protocol": protocol,
                    "reliable": False,
                    "weight": unreliable_weight,
                    "failures": ["missing_protocol"],
                    "o0_median_sec": None,
                    "median_cv_pct": None,
                    "sign_agreement": None,
                    "rank_corr": None,
                    "mean_abs_improvement_pct": None,
                }
            )
            continue
        rows.append(
            evaluate_protocol(
                benchmark_uri,
                protocol,
                protocol_summary,
                min_rank_corr=min_rank_corr,
                min_sign_agreement=min_sign_agreement,
                min_mean_abs_improvement_pct=min_mean_abs_improvement_pct,
                max_median_cv_pct=max_median_cv_pct,
                unreliable_weight=unreliable_weight,
                partial_weight=partial_weight,
            )
        )
    return rows


def reliable_benchmarks(report_rows: Iterable[Dict[str, Any]]) -> Set[str]:
    return {row["benchmark_uri"] for row in report_rows if row.get("reliable")}


def benchmark_weights(report_rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    return {
        row["benchmark_uri"]: float(row.get("weight", 0.0) or 0.0)
        for row in report_rows
    }


def write_report(rows: Sequence[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reliable_benchmarks": sorted(reliable_benchmarks(rows)),
        "benchmark_weights": benchmark_weights(rows),
        "benchmarks": list(rows),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def filter_csv(input_path: Path, output_path: Path, allowlist: Set[str]) -> Tuple[int, int]:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "benchmark_uri" not in fieldnames:
        raise SystemExit(f"{input_path} has no benchmark_uri column")

    kept = [row for row in rows if row.get("benchmark_uri") in allowlist]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(kept), len(rows) - len(kept)


def write_weighted_csv(
    input_path: Path,
    output_path: Path,
    weights: Dict[str, float],
    *,
    weight_col: str = "reward_reliability_weight",
    default_weight: float = 1.0,
) -> Tuple[int, int]:
    """Write all rows with a reward reliability sample-weight column."""
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "benchmark_uri" not in fieldnames:
        raise SystemExit(f"{input_path} has no benchmark_uri column")
    if weight_col not in fieldnames:
        fieldnames.append(weight_col)

    zero_weighted = 0
    for row in rows:
        weight = weights.get(row.get("benchmark_uri", ""), default_weight)
        if weight <= 0.0:
            zero_weighted += 1
        row[weight_col] = f"{weight:.6f}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), zero_weighted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=str(PROJECT_ROOT / "results" / "rf_pilot_summary.json"),
        help="Reward-fidelity summary JSON from scripts/reward_fidelity_pilot.py summarize.",
    )
    parser.add_argument(
        "--protocol",
        default="B",
        choices=["A", "B"],
        help="Protocol to use for the reliability gate. B is the high-fidelity default.",
    )
    parser.add_argument("--min-rank-corr", type=float, default=0.5)
    parser.add_argument("--min-sign-agreement", type=float, default=0.7)
    parser.add_argument("--min-mean-abs-improvement-pct", type=float, default=1.0)
    parser.add_argument("--max-median-cv-pct", type=float, default=5.0)
    parser.add_argument("--unreliable-weight", type=float, default=0.0)
    parser.add_argument("--partial-weight", type=float, default=0.5)
    parser.add_argument(
        "--report-output",
        default=str(PROJECT_ROOT / "results" / "reward_reliability_report.json"),
    )
    parser.add_argument("--input", help="Optional CSV to filter by reliable benchmark_uri.")
    parser.add_argument("--output", help="Output path for the filtered CSV.")
    parser.add_argument(
        "--mode",
        default="filter",
        choices=["filter", "weight"],
        help="filter drops unreliable rows; weight keeps all rows and adds a sample-weight column.",
    )
    parser.add_argument("--weight-col", default="reward_reliability_weight")
    parser.add_argument(
        "--default-weight",
        type=float,
        default=1.0,
        help="Weight for benchmark URIs not present in the pilot summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text())
    report_rows = evaluate_summary(
        summary,
        protocol=args.protocol,
        min_rank_corr=args.min_rank_corr,
        min_sign_agreement=args.min_sign_agreement,
        min_mean_abs_improvement_pct=args.min_mean_abs_improvement_pct,
        max_median_cv_pct=args.max_median_cv_pct,
        unreliable_weight=args.unreliable_weight,
        partial_weight=args.partial_weight,
    )
    write_report(report_rows, Path(args.report_output))
    allowlist = reliable_benchmarks(report_rows)
    weights = benchmark_weights(report_rows)

    print(
        f"Reliable benchmarks under protocol {args.protocol}: "
        f"{len(allowlist)}/{len(report_rows)}"
    )
    for row in report_rows:
        status = "KEEP" if row["reliable"] else "DROP"
        failures = ",".join(row["failures"]) if row["failures"] else "-"
        print(
            f"{status:4s} {row['benchmark']:<14s} "
            f"weight={row['weight']:.2f} "
            f"rank={row['rank_corr'] if row['rank_corr'] is not None else 'NA'} "
            f"sign={row['sign_agreement'] if row['sign_agreement'] is not None else 'NA'} "
            f"meanAbs={row['mean_abs_improvement_pct'] if row['mean_abs_improvement_pct'] is not None else 'NA'} "
            f"cv={row['median_cv_pct'] if row['median_cv_pct'] is not None else 'NA'} "
            f"fail={failures}"
        )

    if args.input or args.output:
        if not args.input or not args.output:
            raise SystemExit("--input and --output must be provided together")
        if args.mode == "filter":
            kept, dropped = filter_csv(Path(args.input), Path(args.output), allowlist)
            print(f"Filtered CSV: kept {kept}, dropped {dropped} -> {args.output}")
        else:
            total, zero_weighted = write_weighted_csv(
                Path(args.input),
                Path(args.output),
                weights,
                weight_col=args.weight_col,
                default_weight=args.default_weight,
            )
            print(
                f"Weighted CSV: wrote {total} rows, "
                f"{zero_weighted} with zero weight -> {args.output}"
            )
    print(f"Saved report to {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
