import json
import logging
import shutil
import re
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if shutil.which("latex"):
    plt.rcParams["text.latex.preamble"] = (
        r"\RequirePackage[tt=false, type1=true]{libertine}"
    )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 12,
            "font.family": "serif",
        }
    )
    logger.info("Using LaTeX font")
else:
    logger.info("LaTeX not found, using default font")


@dataclass(slots=True, frozen=True)
class AccumulatorStat:
    has_tuple: bool
    has_collection: bool
    has_nested_collection: bool
    has_conditional: bool


@dataclass(slots=True, frozen=True)
class Stats:
    status: Literal["Success", "Refuted", "Error"]
    elapsed: float
    input_ast_size: int
    normalizer_ast_size: int
    normalizer: dict
    input_stats: AccumulatorStat | None = None


EVAL_DIR_PATH = Path(__file__).parent
OUTPUT_DIR_PATH = EVAL_DIR_PATH / "output"
CVC5_OUT_PATH = OUTPUT_DIR_PATH / "cvc5"
INK_OUT_PATH = OUTPUT_DIR_PATH / "ink"
PARSYNT_OUT_PATH = OUTPUT_DIR_PATH / "parsynt"
INK_SYNTHESIS_BASELINE_PATHS = {
    "no-decompose": OUTPUT_DIR_PATH / "ink_no_decompose",
    "no-deductive": OUTPUT_DIR_PATH / "ink_no_deductive",
    "no-reduction": OUTPUT_DIR_PATH / "ink_no_reduction",
}
INK_REFUTATION_BASELINE_PATH = OUTPUT_DIR_PATH / "ink_no_refute"
TIMEOUT_SECS = 60 * 10

NUM_NON_HOM = 5
NUM_HOM = 50 - NUM_NON_HOM

NON_HOM_BENCHMARKS = set(
    [
        "ClickstreamAggregate",
        "finance.OHLCAggregator",
        "HourlyAvg",
        "RasterFunction",
        "TelemetryProcessor",
    ]
)

def parse_parsynt_file(filepath):
    """
    Parse a single parsynt output file and extract elapsed time.
    Returns (program_name, elapsed_time) where elapsed_time is None if invalid.
    """
    program_name = Path(filepath).stem  # filename without extension
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return program_name, None
    
    # Check for errors or fatal errors (case insensitive)
    if re.search(r'\b(error|fatal)\b', content, re.IGNORECASE):
        return program_name, None
    
    # Check for empty join function
    if '?? = (empty);' in content or '{ ;' in content:
        return program_name, None
    
    # Look for synthesis time pattern
    pattern = r'// Synthesized in \(predicate : ([0-9]+\.[0-9]+) s\) \+ \(join : ([0-9]+\.[0-9]+) s\)'
    match = re.search(pattern, content)
    
    if match:
        predicate_time = float(match.group(1))
        join_time = float(match.group(2))
        elapsed_time = predicate_time + join_time
        return program_name, elapsed_time
    
    return program_name, None


def load_parsynt_data() -> pd.DataFrame:
    parsynt_data = {}
    parsynt_dir = PARSYNT_OUT_PATH

    for filepath in parsynt_dir.glob("*.txt"):
        program_name, elapsed_time = parse_parsynt_file(filepath)
        parsynt_data[program_name] = elapsed_time

    # Create DataFrame
    df = pd.DataFrame.from_dict(parsynt_data, orient="index", columns=["Parsynt"])
    df.index.name = "Program"
    return df


def postprocess_stats(stats: Stats) -> Stats:
    if stats.status in ["Error", "Failure"]:
        return replace(stats, status="Timeout", elapsed=TIMEOUT_SECS)

    if stats.elapsed > TIMEOUT_SECS + 60:
        return replace(stats, status="Timeout", elapsed=TIMEOUT_SECS)

    normalizer = stats.normalizer["Ok"]
    return replace(stats, normalizer=normalizer)


def load_data_from_dir(out_path: Path) -> dict[str, Stats]:
    prog_stats: dict[str, Stats] = {}

    jsons = list(out_path.glob("*.json"))
    for json_p in jsons:
        prog_name = json_p.stem
        with open(json_p, "r") as f:
            try:
                s = Stats(**json.load(f))
                prog_stats[prog_name] = postprocess_stats(s)
            except json.JSONDecodeError:
                logger.error(f"Error decoding {json_p}")
                prog_stats[prog_name] = Stats("Error", TIMEOUT_SECS, 0, 0, {})
    return prog_stats


def report_data(df, variant: str):
    if df.empty:
        logger.error(f"{variant} is empty")
        return

    df_solved = df[df["status"] == "Success"]
    num_solved = df_solved.shape[0]
    pct_solved = 100 * num_solved / NUM_HOM
    avg_time = df_solved["elapsed"].mean()
    logger.info(
        f"{variant} solved: {num_solved} / {NUM_HOM} ({pct_solved:.2f}%) within average time {avg_time:.2f}s"
    )

    df_refuted = df[df["status"] == "Refuted"]
    num_refuted = df_refuted.shape[0]
    pct_refuted = 100 * num_refuted / NUM_NON_HOM
    avg_time = df_refuted["elapsed"].mean()
    logger.info(
        f"{variant} refuted: {num_refuted} / {NUM_NON_HOM} ({pct_refuted:.2f}%) within average time {avg_time:.2f}s"
    )


def plot_baseline_cdf(
    df,
    num_benchmarks,
    order,
    labels,
    title=None,
    pdf_path=None,
    *,
    ylabel=r"\% of Benchmarks Solved",
    xmax=TIMEOUT_SECS + 1,
    xmin=1,
    use_log_scale: bool=False,
    legend_loc="center right",
    legend_ncols=2,
    bbox_to_anchor=None,
    use_survival_plot: bool=False,
):
    df = (
        df.melt(id_vars="Program", var_name="variant")
        .fillna(1e200)
        .sort_values(by=["variant", "value"])
    )

    # Make time values cumulative within each variant
    if use_survival_plot:
        df["value"] = df.groupby("variant")["value"].cumsum()
    df["cumulative_count"] = df.groupby("variant").cumcount() + 1
    df["percentage"] = df["cumulative_count"].apply(lambda x: 100 * x / num_benchmarks)

    plt.figure(figsize=(5, 3))

    sns_plot = sns.lineplot(
        data=df,
        x="value",
        y="percentage",
        hue="variant",
        style="variant",
        markers=True,
        dashes=False,
        errorbar=None,
        hue_order=order,
    )
    plt.grid(True)
    plt.xticks(np.arange(0, xmax, xmax // 5))
    plt.ylim(0, 101)
    plt.xlim(xmin, xmax)
    plt.xlabel("Running Total (sec)" if use_survival_plot else "Time (sec)")
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    if use_log_scale:
        plt.xscale("log")

    handles, _ = sns_plot.get_legend_handles_labels()
    sns_plot.legend(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        ncols=legend_ncols,
        columnspacing=0.8,
        bbox_to_anchor=bbox_to_anchor,
    )

    if pdf_path is not None:
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")


def to_time_df(data: dict[str, Stats], variant: str, use_hom: bool) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data, orient="index")

    if use_hom:
        df = df.loc[~df.index.isin(NON_HOM_BENCHMARKS)]
    else:
        df = df.loc[df.index.isin(NON_HOM_BENCHMARKS)]

    df.loc[~df["status"].isin(["Success", "Refuted"]), "elapsed"] = None
    df = df[["elapsed"]].rename(columns={"elapsed": variant})
    return df


def to_ast_size_df(data: dict[str, Stats]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data, orient="index")
    return df[["input_ast_size", "normalizer_ast_size"]]


def to_accumulator_stat_df(data: dict[str, Stats]) -> pd.DataFrame:
    data = {k: v.input_stats for k, v in data.items()}
    df = pd.DataFrame.from_dict(data, orient="index")
    return df.astype(float)


def main():
    ink_data = load_data_from_dir(INK_OUT_PATH)
    cvc5_data = load_data_from_dir(CVC5_OUT_PATH)

    logger.info("Section 6. Figure 15: Benchmark statistics (size of ASTs)")
    ast_size_df = to_ast_size_df(ink_data)
    print(
        tabulate(
            ast_size_df.describe().T[["mean", "50%", "max"]].T, 
            headers="keys", 
            tablefmt="psql", 
            floatfmt=".2f"
        )
    )
    print("=" * 80 + "\n\n")

    logger.info("Section 6. Figure 16: Benchmark statistics (% of accumulators types)")
    ink_stats_df = to_accumulator_stat_df(ink_data).mean()
    print(
        tabulate(
            ink_stats_df.to_frame().T, headers="keys", tablefmt="psql", floatfmt=".1%"
        )
    )
    print("=" * 80 + "\n\n")

    ink_df = to_time_df(ink_data, "Ink", use_hom=True)
    cvc5_df = to_time_df(cvc5_data, "CVC5", use_hom=True)
    parsynt_df = load_parsynt_data()
    baseline_df = ink_df.join(pd.concat([parsynt_df, cvc5_df], axis=1), how="outer")

    logger.info("Section 6.1. Average time and # solved for each baseline")
    baseline_stats = baseline_df.describe().T[["mean", "count"]].rename(
        columns={"mean": "Average Time", "count": "Number Solved"}
    )
    baseline_stats["% of Number Solved"] = baseline_stats["Number Solved"] / NUM_HOM * 100
    print(
        tabulate(
            baseline_stats,
            headers="keys",
            tablefmt="psql",
            floatfmt=".2f",
        )
    )
    print("=" * 80 + "\n\n")

    both_cvc5_parsynt_solved = baseline_df[["Parsynt", "CVC5"]].dropna()
    logger.info("Section 6.1. Average time for benchmarks solved by both Parsynt and CVC5")
    print(
        tabulate(
            both_cvc5_parsynt_solved.describe().T[["mean", "count"]],
            headers="keys",
            tablefmt="psql",
            floatfmt=".2f",
        )
    )
    print("=" * 80 + "\n\n")

    both_ink_and_cvc5_solved = baseline_df[["Ink", "CVC5"]].dropna()
    logger.info("Section 6.1. Average time for benchmarks solved by both Ink and CVC5")
    print(
        tabulate(
            both_ink_and_cvc5_solved.describe().T[["mean", "count"]],
            headers="keys",
            tablefmt="psql",
            floatfmt=".2f",
        )
    )
    print("=" * 80 + "\n\n")

    both_ink_and_parsynt_solved = baseline_df[["Ink", "Parsynt"]].dropna()
    logger.info("Section 6.1. Average time for benchmarks solved by both Ink and Parsynt")
    print(
        tabulate(
            both_ink_and_parsynt_solved.describe().T[["mean", "count"]],
            headers="keys",
            tablefmt="psql",
            floatfmt=".2f",
        )
    )
    print("=" * 80 + "\n\n")

    ink_avg_time = ink_df["Ink"].mean()
    ink_num_solved = ink_df["Ink"].count()
    cvc5_num_solved = cvc5_df["CVC5"].count()
    parsynt_num_solved = parsynt_df["Parsynt"].count()

    print(f"Result for RQ1: Among the {NUM_HOM} homomorphic UDAFs, Ink can successfully synthesize merge operators for {ink_num_solved} of them "
          f"({ink_num_solved / NUM_HOM * 100:.1f}%), taking {ink_avg_time:.2f} seconds per benchmark on average. "
          f"In comparison, the two baselines (CVC5 and Parsynt) synthesize merge operators for {parsynt_num_solved} ({parsynt_num_solved / NUM_HOM * 100:.1f}%) and "
          f"{cvc5_num_solved} ({cvc5_num_solved / NUM_HOM * 100:.1f}%) of the same benchmarks, respectively.")
    print("=" * 80 + "\n\n")

    # baseline cdf plot
    baseline_cdf_df = baseline_df.reset_index().rename(columns={"index": "Program"})
    baseline_cdf_df = pd.concat(
        [
            baseline_cdf_df,
            pd.DataFrame(
                [["q-last", 1e200, 1e200, 1e200]], columns=baseline_cdf_df.columns
            ),
        ],
        axis=0,
    )
    order = ["Ink", "CVC5", "Parsynt"]
    labels = [r"\textsc{Ink}", r"\textsc{CVC5}", r"\textsc{Parsynt}"]
    plot_baseline_cdf(
        baseline_cdf_df,
        NUM_HOM,
        order,
        labels,
        pdf_path="baseline-cdf.pdf",
        legend_loc="best",
        legend_ncols=1,
        bbox_to_anchor=(1, 0.95),
    )
    logger.info(
        "Section 6.1. Figure 17: CDF of merge operator synthesis -- saved to baseline-cdf.pdf"
    )
    print("=" * 80 + "\n\n")

    # refutation baseline stats
    logger.info("Section 6.2. Refutation baseline statistics")
    ink_ref_time_df = to_time_df(ink_data, "Ink", use_hom=False)
    cvc5_ref_time_df = to_time_df(cvc5_data, "CVC5", use_hom=False)
    df = (
        ink_ref_time_df.join(cvc5_ref_time_df, how="outer")
        .reset_index()
        .rename(columns={"index": "Program"})
    ).describe().T[["mean", "50%", "count"]].T
    ink_ref_num_solved = ink_ref_time_df["Ink"].count()
    cvc5_ref_num_solved = cvc5_ref_time_df["CVC5"].count()
    print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".2f"))
    print(f"Result for RQ2: Ink is able to refute the existence of a merge operator for {ink_ref_num_solved} "
          f"non-homomorphic UDAFs, whereas CVC5 refutes {cvc5_ref_num_solved}.")
    print("=" * 80 + "\n\n")

    # ablation cdf plot
    ablation_dfs = [
        to_time_df(load_data_from_dir(path), baseline, use_hom=True)
        for baseline, path in INK_SYNTHESIS_BASELINE_PATHS.items()
    ]
    ablation_df = pd.concat(ablation_dfs, axis=1)
    logger.info("Section 6.3. Ablation statistics")
    print(
        tabulate(
            ablation_df.describe().T[["count", "mean"]],
            headers="keys",
            tablefmt="psql",
            floatfmt=".2f",
        )
    )
    num_ink_solved = ink_df["Ink"].count()
    ink_time = ink_df["Ink"].mean()
    ablation_pcts = []
    ablation_time_multipliers = []
    for ablation in INK_SYNTHESIS_BASELINE_PATHS:
        num_solved = ablation_df[ablation].count()
        avg_time = ablation_df[ablation].mean()
        pct_fewer = 100 * (num_ink_solved - num_solved) / num_ink_solved
        time_multiplier = avg_time / ink_time
        ablation_pcts.append(pct_fewer)
        ablation_time_multipliers.append(time_multiplier)
        print(
            f"Ablation {ablation}: solved {num_solved} ({num_solved / NUM_HOM * 100:.2f}%) benchmarks in {avg_time:.2f}s \t\t {pct_fewer:.2f}% fewer than Ink in {time_multiplier:.2f}x time"
        )

    print(
        "RQ3: Without deductive synthesis and decomposition, the performance of Ink degrades considerably, "
        f"solving {min(ablation_pcts):.2f}-{max(ablation_pcts):.2f}% fewer benchmarks and taking {min(ablation_time_multipliers):.2f}-{max(ablation_time_multipliers):.2f}x as long to solve a benchmark on average."
    )
    print("=" * 80 + "\n\n")

    df = (
        ink_df.join(ablation_df, how="outer")
        .reset_index()
        .rename(columns={"index": "Program"})
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame([["q-last", 1e200, 1e200, 1e200, 1e200]], columns=df.columns),
        ],
        axis=0,
    )
    order = ["Ink", "no-reduction", "no-deductive", "no-decompose"]
    labels = [
        r"\textsc{Ink}",
        r"\textsc{NoReduce}",
        r"\textsc{NoDeduce}",
        r"\textsc{NoDecomp}",
    ]
    plot_baseline_cdf(
        df,
        NUM_HOM,
        order,
        labels,
        pdf_path="ablation-cdf.pdf",
        legend_loc="center right",
        legend_ncols=2,
    )
    logger.info(
        "Section 5.3. Figure 18: CDF of merge operator synthesis -- saved to baseline-cdf.pdf"
    )
    print("=" * 80 + "\n\n")

    # refutation ablation stats
    logger.info("Section 6.4. Refutation ablation statistics")
    ablation_ref_df = to_time_df(
        load_data_from_dir(INK_REFUTATION_BASELINE_PATH), "no-refute", use_hom=False
    )
    df = (
        ink_ref_time_df.join(ablation_ref_df, how="outer")
        .reset_index()
        .rename(columns={"index": "Program"})
    )
    print(tabulate(df.describe(), headers="keys", tablefmt="psql", floatfmt=".2f"))
    ink_ref_avg_time = ink_ref_time_df["Ink"].mean()
    ink_ref_num_solved = ink_ref_time_df["Ink"].count()
    no_refute_avg_time = ablation_ref_df["no-refute"].mean()
    no_refute_num_solved = ablation_ref_df["no-refute"].count()

    print("The ablation of Ink that does not leverage the refutation rules fails to "
        f"refute {NUM_NON_HOM - no_refute_num_solved} of the {NUM_NON_HOM} non-homomorphic benchmarks and takes {no_refute_avg_time / ink_ref_avg_time:.2f}x as long.")


if __name__ == "__main__":
    main()
