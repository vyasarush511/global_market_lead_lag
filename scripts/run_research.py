"""Run the full global market lead-lag research pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from global_market_lead_lag.config import MarketConfig  # noqa: E402
from global_market_lead_lag.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2010-01-01", help="First download date.")
    parser.add_argument("--end-date", default=None, help="Exclusive end date. Defaults to latest available.")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated tables and figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MarketConfig(start_date=args.start_date, end_date=args.end_date)
    outputs = run_pipeline(config=config, output_dir=ROOT / args.output_dir, save_outputs=True)

    print("\nBaseline regression")
    print(outputs["baseline_regression"])
    print("\nStrategy variants")
    print(outputs["variant_results"])
    print("\nTrain/validation/test selected strategy")
    print(outputs["tvt"]["selected"])
    print(outputs["tvt"]["test_results"])
    print("\nWalk-forward summary")
    print(outputs["wf_summary"])
    print(f"\nWrote reports to: {ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
