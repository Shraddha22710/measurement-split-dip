from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    rows = []
    for path in sorted(root.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["run"] = path.parent.name
        rows.append(row)

    if not rows:
        raise SystemExit(f"No summary.json files found under {root}")

    df = pd.DataFrame(rows)
    keep = [
        "run",
        "operator",
        "split_domain",
        "oracle_psnr",
        "val_psnr",
        "smooth_val_psnr",
        "final_psnr",
        "oracle_iteration",
        "val_iteration",
        "smooth_val_iteration",
        "val_oracle_gap",
        "smooth_val_oracle_gap",
    ]
    keep = [c for c in keep if c in df.columns]
    out = root / "aggregate_summary.csv"
    df[keep].sort_values(["operator", "split_domain", "run"]).to_csv(out, index=False)

    printable = df[keep].copy()
    for col in printable.columns:
        if printable[col].dtype.kind == "f":
            printable[col] = printable[col].map(lambda x: f"{x:.3f}")
    print(printable.sort_values(["operator", "split_domain", "run"]).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
