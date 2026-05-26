"""
5_generate_tables.py
───────────────────────────────────────────────
Script to generate a markdown table summarizing the unlearning results
from results/cfk_metrics.json and results/euk_metrics.json.
"""

import json
import argparse
from pathlib import Path
from utils.config import RESULTS_PATH

def format_acc(mean, std):
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"

def format_time(mean, std):
    return f"{mean:.2f}s ± {std:.2f}s"

def format_ratio(mean, std):
    return f"{mean:.4f} ± {std:.4f}"

def format_rk(mean, std):
    if mean is None or std is None:
        return "NaN"
    if isinstance(mean, str) and mean.lower() == "nan":
        return "NaN"
    if isinstance(std, str) and std.lower() == "nan":
        return "NaN"
    try:
        import math
        f_mean = float(mean)
        f_std = float(std)
        if math.isnan(f_mean) or math.isnan(f_std):
            return "NaN"
        return f"{f_mean:.4f} ± {f_std:.4f}"
    except (ValueError, TypeError):
        return "NaN"

def main():
    parser = argparse.ArgumentParser(description="Generate a markdown table of results.")
    parser.add_argument("--cfk_file", type=str, default=str(RESULTS_PATH / "cfk_metrics.json"),
                        help="Path to CFK metrics JSON file")
    parser.add_argument("--euk_file", type=str, default=str(RESULTS_PATH / "euk_metrics.json"),
                        help="Path to EUK metrics JSON file")
    parser.add_argument("--output_file", type=str, default=str(RESULTS_PATH / "metrics_table.md"),
                        help="Path to save the generated Markdown table")
    args = parser.parse_args()

    cfk_path = Path(args.cfk_file)
    euk_path = Path(args.euk_file)

    if not cfk_path.exists():
        print(f"Error: {cfk_path} not found.")
        return
    if not euk_path.exists():
        print(f"Error: {euk_path} not found.")
        return

    with open(cfk_path, "r", encoding="utf-8") as f:
        cfk_data = json.load(f)
    with open(euk_path, "r", encoding="utf-8") as f:
        euk_data = json.load(f)

    cfk_agg = cfk_data["aggregated"]
    euk_agg = euk_data["aggregated"]

    table_data = [
        {
            "name": "**Base Model**",
            "retain": format_acc(cfk_agg["base_retain"]["mean"], cfk_agg["base_retain"]["std"]),
            "forget": format_acc(cfk_agg["base_forget"]["mean"], cfk_agg["base_forget"]["std"]),
            "test": format_acc(cfk_agg["base_test"]["mean"], cfk_agg["base_test"]["std"]),
            "time": format_time(cfk_agg["base_time"]["mean"], cfk_agg["base_time"]["std"]),
            "rr": format_ratio(cfk_agg["base_RR"]["mean"], cfk_agg["base_RR"]["std"]),
            "tr": format_ratio(cfk_agg["base_TR"]["mean"], cfk_agg["base_TR"]["std"]),
            "rk": format_rk(cfk_agg["base_RK"]["mean"], cfk_agg["base_RK"]["std"]),
        },
        {
            "name": "**Naive Model**",
            "retain": format_acc(cfk_agg["naive_retain"]["mean"], cfk_agg["naive_retain"]["std"]),
            "forget": format_acc(cfk_agg["naive_forget"]["mean"], cfk_agg["naive_forget"]["std"]),
            "test": format_acc(cfk_agg["naive_test"]["mean"], cfk_agg["naive_test"]["std"]),
            "time": format_time(cfk_agg["naive_time"]["mean"], cfk_agg["naive_time"]["std"]),
            "rr": format_ratio(cfk_agg["naive_RR"]["mean"], cfk_agg["naive_RR"]["std"]),
            "tr": format_ratio(cfk_agg["naive_TR"]["mean"], cfk_agg["naive_TR"]["std"]),
            "rk": format_rk(cfk_agg["naive_RK"]["mean"], cfk_agg["naive_RK"]["std"]),
        },
        {
            "name": "**CFK Unlearning**",
            "retain": format_acc(cfk_agg["unlearned_retain"]["mean"], cfk_agg["unlearned_retain"]["std"]),
            "forget": format_acc(cfk_agg["unlearned_forget"]["mean"], cfk_agg["unlearned_forget"]["std"]),
            "test": format_acc(cfk_agg["unlearned_test"]["mean"], cfk_agg["unlearned_test"]["std"]),
            "time": format_time(cfk_agg["unlearned_time"]["mean"], cfk_agg["unlearned_time"]["std"]),
            "rr": format_ratio(cfk_agg["unlearned_RR"]["mean"], cfk_agg["unlearned_RR"]["std"]),
            "tr": format_ratio(cfk_agg["unlearned_TR"]["mean"], cfk_agg["unlearned_TR"]["std"]),
            "rk": format_rk(cfk_agg["unlearned_RK"]["mean"], cfk_agg["unlearned_RK"]["std"]),
        },
        {
            "name": "**EUK Unlearning**",
            "retain": format_acc(euk_agg["unlearned_retain"]["mean"], euk_agg["unlearned_retain"]["std"]),
            "forget": format_acc(euk_agg["unlearned_forget"]["mean"], euk_agg["unlearned_forget"]["std"]),
            "test": format_acc(euk_agg["unlearned_test"]["mean"], euk_agg["unlearned_test"]["std"]),
            "time": format_time(euk_agg["unlearned_time"]["mean"], euk_agg["unlearned_time"]["std"]),
            "rr": format_ratio(euk_agg["unlearned_RR"]["mean"], euk_agg["unlearned_RR"]["std"]),
            "tr": format_ratio(euk_agg["unlearned_TR"]["mean"], euk_agg["unlearned_TR"]["std"]),
            "rk": format_rk(euk_agg["unlearned_RK"]["mean"], euk_agg["unlearned_RK"]["std"]),
        }
    ]

    # Generate Markdown table
    md_lines = []
    md_lines.append("| Model / Protocol | Retain Accuracy (%) | Forget Accuracy (%) | Test Accuracy (%) | Training Time (s) | Retain Ratio (RR) | Time Ratio (TR) | Residual Knowledge (RK) |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in table_data:
        md_lines.append(
            f"| {row['name']} | {row['retain']} | {row['forget']} | {row['test']} | {row['time']} | {row['rr']} | {row['tr']} | {row['rk']} |"
        )

    md_table = "\n".join(md_lines) + "\n"

    # Write to output file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print(f"Table successfully saved to {output_path}:\n")
    print(md_table)

if __name__ == "__main__":
    main()
