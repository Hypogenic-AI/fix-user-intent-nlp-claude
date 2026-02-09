"""Analysis and visualization of experiment results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

from config import DATA_DIR, PLOTS_DIR, FIGURES_DIR, MODELS, CORRECTION_PROMPTS

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150


def load_results():
    with open(os.path.join(DATA_DIR, "experiment1_results.json")) as f:
        exp1 = json.load(f)
    with open(os.path.join(DATA_DIR, "experiment2_results.json")) as f:
        exp2 = json.load(f)
    with open(os.path.join(DATA_DIR, "experiment3_results.json")) as f:
        exp3 = json.load(f)
    return exp1, exp2, exp3


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def analyze_experiment_1(exp1):
    """Analyze intent violation rates across models and prompt strategies."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 ANALYSIS: INTENT VIOLATION RATES")
    print("=" * 70)

    results_table = {}

    for model in MODELS.keys():
        for prompt in CORRECTION_PROMPTS.keys():
            subset = [r for r in exp1 if r["model"] == model and r["prompt_strategy"] == prompt]
            if not subset:
                continue

            n = len(subset)
            shifts = [1 if r["intent_shifted"] else 0 for r in subset]
            shift_rate = np.mean(shifts)
            ci_low, ci_high = bootstrap_ci(shifts)

            edit_ratios = [r["edit_ratio"] for r in subset]
            sem_sims = [r["semantic_similarity"] for r in subset]
            nli_fwd = [r["nli_forward"] for r in subset]
            nli_bwd = [r["nli_backward"] for r in subset]

            # Unchanged queries (edit_ratio == 0)
            unchanged = sum(1 for r in subset if r["edit_ratio"] == 0.0)
            unchanged_pct = unchanged / n

            results_table[(model, prompt)] = {
                "n": n,
                "shift_rate": shift_rate,
                "shift_ci": (ci_low, ci_high),
                "edit_ratio_mean": np.mean(edit_ratios),
                "edit_ratio_std": np.std(edit_ratios),
                "sem_sim_mean": np.mean(sem_sims),
                "sem_sim_std": np.std(sem_sims),
                "nli_fwd_mean": np.mean(nli_fwd),
                "nli_bwd_mean": np.mean(nli_bwd),
                "unchanged_pct": unchanged_pct,
            }

            print(f"\n  {model} / {prompt} (n={n}):")
            print(f"    Intent shift rate: {shift_rate:.1%} [{ci_low:.1%}, {ci_high:.1%}]")
            print(f"    Unchanged queries: {unchanged_pct:.1%}")
            print(f"    Edit ratio:        {np.mean(edit_ratios):.3f} +/- {np.std(edit_ratios):.3f}")
            print(f"    Semantic sim:      {np.mean(sem_sims):.3f} +/- {np.std(sem_sims):.3f}")
            print(f"    NLI forward:       {np.mean(nli_fwd):.3f}")
            print(f"    NLI backward:      {np.mean(nli_bwd):.3f}")

    # By dataset
    print("\n\n--- By Dataset ---")
    for dataset in ["banking77", "clinc150"]:
        subset = [r for r in exp1 if r["dataset"] == dataset]
        shifts = [1 if r["intent_shifted"] else 0 for r in subset]
        print(f"  {dataset}: shift_rate={np.mean(shifts):.1%} (n={len(subset)})")

    # Statistical comparison: fix_errors vs rewrite_clearly vs improve
    print("\n\n--- Statistical Comparisons (Wilcoxon signed-rank) ---")
    for model in MODELS.keys():
        prompts = list(CORRECTION_PROMPTS.keys())
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                p1, p2 = prompts[i], prompts[j]
                s1 = [1 if r["intent_shifted"] else 0
                      for r in exp1 if r["model"] == model and r["prompt_strategy"] == p1]
                s2 = [1 if r["intent_shifted"] else 0
                      for r in exp1 if r["model"] == model and r["prompt_strategy"] == p2]
                if len(s1) == len(s2) and len(s1) > 0:
                    try:
                        stat, p_val = stats.wilcoxon(s1, s2)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"  {model}: {p1} vs {p2}: W={stat:.0f}, p={p_val:.4f} {sig}")
                    except ValueError:
                        print(f"  {model}: {p1} vs {p2}: Cannot compute (identical distributions)")

    return results_table


def analyze_experiment_2(exp2):
    """Analyze metric correlation with LLM-as-judge."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 ANALYSIS: METRIC VALIDATION")
    print("=" * 70)

    valid = [r for r in exp2 if r["judge_label"] in ("preserved", "changed")]
    print(f"\n  Valid judge labels: {len(valid)}/{len(exp2)}")

    if len(valid) < 10:
        print("  Not enough valid judge labels for analysis")
        return {}

    # Binary: judge says preserved (1) or changed (0)
    judge_binary = [1 if r["judge_label"] == "preserved" else 0 for r in valid]

    # Automated metrics
    classifier_binary = [0 if r["intent_shifted"] else 1 for r in valid]
    sem_sims = [r["semantic_similarity"] for r in valid]
    edit_ratios = [r["edit_ratio"] for r in valid]
    nli_fwd = [r["nli_forward"] for r in valid]
    nli_bwd = [r["nli_backward"] for r in valid]
    nli_bid = [r["nli_bidirectional"] for r in valid]

    correlations = {}

    metrics = {
        "intent_classifier": classifier_binary,
        "semantic_similarity": sem_sims,
        "edit_ratio_inv": [1 - e for e in edit_ratios],
        "nli_forward": nli_fwd,
        "nli_backward": nli_bwd,
        "nli_bidirectional": nli_bid,
    }

    for name, values in metrics.items():
        # Point-biserial correlation for binary judge vs continuous metric
        if name == "intent_classifier":
            # Cohen's kappa for agreement between two binary classifiers
            from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
            kappa = cohen_kappa_score(judge_binary, values)
            acc = accuracy_score(judge_binary, values)
            f1 = f1_score(judge_binary, values)
            print(f"\n  {name}:")
            print(f"    Cohen's kappa: {kappa:.3f}")
            print(f"    Accuracy: {acc:.1%}")
            print(f"    F1: {f1:.3f}")
            correlations[name] = {"kappa": kappa, "accuracy": acc, "f1": f1}
        else:
            r_val, p_val = stats.pointbiserialr(judge_binary, values)
            print(f"\n  {name}:")
            print(f"    Point-biserial r: {r_val:.3f}, p={p_val:.4f}")
            correlations[name] = {"r": r_val, "p": p_val}

    # Agreement matrix
    print("\n\n--- Judge vs Classifier Agreement ---")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(judge_binary, classifier_binary, labels=[0, 1])
    print(f"  Confusion matrix (rows=judge, cols=classifier):")
    print(f"    [changed, preserved]")
    print(f"    Changed:   {cm[0]}")
    print(f"    Preserved: {cm[1]}")

    return correlations


def analyze_experiment_3(exp3):
    """Analyze confidence-aware strategy vs baselines."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 ANALYSIS: CONFIDENCE-AWARE STRATEGY")
    print("=" * 70)

    ca = exp3["confidence_aware"]
    ac = exp3["always_correct"]
    conf_dist = exp3["confidence_distribution"]

    # Confidence-aware results
    ca_shifts = sum(1 for r in ca if r.get("intent_shifted", False))
    ca_clarify = sum(1 for r in ca if r["action"] == "clarify")
    ca_abstain = sum(1 for r in ca if r["action"] == "abstain")
    ca_correct = sum(1 for r in ca if r["action"] == "auto_correct")
    n_ca = len(ca)

    # Always-correct results
    ac_shifts = sum(1 for r in ac if r.get("intent_shifted", False))
    n_ac = len(ac)

    # No-action baseline: 0 shifts by definition
    print(f"\n  Dataset size: {n_ca}")
    print(f"\n  Confidence distribution:")
    print(f"    High (>{0.8}):  {conf_dist['high']}/{n_ca} ({conf_dist['high']/n_ca:.1%})")
    print(f"    Medium:     {conf_dist['medium']}/{n_ca} ({conf_dist['medium']/n_ca:.1%})")
    print(f"    Low (<{0.4}):   {conf_dist['low']}/{n_ca} ({conf_dist['low']/n_ca:.1%})")

    print(f"\n  Strategy Comparison:")
    print(f"  {'Strategy':<25} {'Intent Shifts':>15} {'Clarifications':>15} {'Abstentions':>12}")
    print(f"  {'-'*67}")

    strategies = {
        "Confidence-Aware": (ca_shifts, n_ca, ca_clarify, ca_abstain),
        "Always-Correct": (ac_shifts, n_ac, 0, 0),
        "No-Action": (0, n_ca, 0, n_ca),
        "Always-Clarify": (0, n_ca, n_ca, 0),
    }

    for name, (shifts, n, clarify, abstain) in strategies.items():
        shift_str = f"{shifts}/{n} ({shifts/n:.1%})"
        clarify_str = f"{clarify}/{n} ({clarify/n:.1%})" if clarify > 0 else "0"
        abstain_str = f"{abstain}/{n} ({abstain/n:.1%})" if abstain > 0 else "0"
        print(f"  {name:<25} {shift_str:>15} {clarify_str:>15} {abstain_str:>12}")

    # Effective accuracy: penalizes both intent shifts AND excessive clarification
    # EA = 1 - shift_rate - 0.3 * clarify_rate (where 0.3 is annoyance weight)
    annoyance_weight = 0.3
    for name, (shifts, n, clarify, abstain) in strategies.items():
        ea = 1.0 - (shifts / n) - annoyance_weight * (clarify / n)
        print(f"  {name}: Effective Accuracy = {ea:.3f}")

    # Statistical test: confidence-aware vs always-correct
    ca_shift_vec = [1 if r.get("intent_shifted", False) else 0 for r in ca]
    ac_shift_vec = [1 if r.get("intent_shifted", False) else 0 for r in ac]

    if len(ca_shift_vec) == len(ac_shift_vec):
        stat, p_val = stats.wilcoxon(ca_shift_vec, ac_shift_vec, alternative="less")
        print(f"\n  Wilcoxon test (CA < AC): W={stat:.0f}, p={p_val:.4f}")
    else:
        stat, p_val = stats.mannwhitneyu(ca_shift_vec, ac_shift_vec, alternative="less")
        print(f"\n  Mann-Whitney test (CA < AC): U={stat:.0f}, p={p_val:.4f}")

    # Clarification quality examples
    clarify_examples = [r for r in ca if r["action"] == "clarify"][:5]
    if clarify_examples:
        print("\n  Example Clarifications:")
        for ex in clarify_examples:
            print(f"    Query: \"{ex['original']}\"")
            print(f"    Clarification: \"{ex['response']}\"")
            print(f"    Confidence: {ex['confidence']:.2f}")
            print()

    return strategies


def create_visualizations(exp1, exp2, exp3, results_table):
    """Create all visualizations."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Figure 1: Intent Violation Rate by Model and Strategy ──
    fig, ax = plt.subplots(figsize=(12, 6))
    models = list(MODELS.keys())
    prompts = list(CORRECTION_PROMPTS.keys())
    prompt_labels = {"fix_errors": "Fix Errors", "rewrite_clearly": "Rewrite Clearly", "improve": "Improve"}

    x = np.arange(len(models))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#F44336"]

    for i, prompt in enumerate(prompts):
        rates = []
        ci_lows = []
        ci_highs = []
        for model in models:
            if (model, prompt) in results_table:
                r = results_table[(model, prompt)]
                rates.append(r["shift_rate"] * 100)
                ci_lows.append(r["shift_ci"][0] * 100)
                ci_highs.append(r["shift_ci"][1] * 100)
            else:
                rates.append(0)
                ci_lows.append(0)
                ci_highs.append(0)

        yerr_low = [r - c for r, c in zip(rates, ci_lows)]
        yerr_high = [c - r for r, c in zip(rates, ci_highs)]

        bars = ax.bar(x + i * width, rates, width, label=prompt_labels[prompt],
                      color=colors[i], alpha=0.85, edgecolor="white")
        ax.errorbar(x + i * width, rates, yerr=[yerr_low, yerr_high],
                    fmt="none", color="black", capsize=4)

    ax.set_xlabel("Model")
    ax.set_ylabel("Intent Violation Rate (%)")
    ax.set_title("Intent Violation Rate by Model and Correction Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend(title="Strategy")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_intent_violation_rates.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_intent_violation_rates.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 2: Metrics Distribution by Intent Shift ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    shifted = [r for r in exp1 if r["intent_shifted"]]
    preserved = [r for r in exp1 if not r["intent_shifted"]]

    metric_pairs = [
        ("semantic_similarity", "Semantic Similarity", axes[0]),
        ("edit_ratio", "Edit Ratio", axes[1]),
        ("nli_forward", "NLI Forward Entailment", axes[2]),
    ]

    for key, label, ax in metric_pairs:
        s_vals = [r[key] for r in shifted]
        p_vals = [r[key] for r in preserved]

        ax.hist(p_vals, bins=30, alpha=0.6, color="#4CAF50", label="Preserved", density=True)
        ax.hist(s_vals, bins=30, alpha=0.6, color="#F44336", label="Shifted", density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()

    plt.suptitle("Metric Distributions: Preserved vs Shifted Intent", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_metric_distributions.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_metric_distributions.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 3: Edit Ratio vs Semantic Similarity scatter ──
    fig, ax = plt.subplots(figsize=(10, 7))
    shifted_mask = [r["intent_shifted"] for r in exp1]
    colors_scatter = ["#F44336" if s else "#4CAF50" for s in shifted_mask]
    edit_ratios = [r["edit_ratio"] for r in exp1]
    sem_sims = [r["semantic_similarity"] for r in exp1]

    ax.scatter(edit_ratios, sem_sims, c=colors_scatter, alpha=0.4, s=20)
    ax.set_xlabel("Edit Ratio (higher = more changes)")
    ax.set_ylabel("Semantic Similarity")
    ax.set_title("Edit Ratio vs Semantic Similarity\n(Green = Intent Preserved, Red = Intent Shifted)")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.6, label="Intent Preserved"),
        Patch(facecolor="#F44336", alpha=0.6, label="Intent Shifted"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_edit_vs_similarity.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_edit_vs_similarity.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 4: Experiment 3 - Strategy Comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ca = exp3["confidence_aware"]
    ac = exp3["always_correct"]
    n = len(ca)

    ca_shifts = sum(1 for r in ca if r.get("intent_shifted", False))
    ac_shifts = sum(1 for r in ac if r.get("intent_shifted", False))
    ca_clarify = sum(1 for r in ca if r["action"] == "clarify")

    strategies_names = ["Always\nCorrect", "Confidence\nAware", "No\nAction", "Always\nClarify"]
    shift_rates = [ac_shifts / n * 100, ca_shifts / n * 100, 0, 0]
    clarify_rates = [0, ca_clarify / n * 100, 0, 100]
    colors_bars = ["#F44336", "#2196F3", "#9E9E9E", "#FF9800"]

    # Left panel: Intent shift rates
    bars = axes[0].bar(strategies_names, shift_rates, color=colors_bars, alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Intent Violation Rate (%)")
    axes[0].set_title("Intent Violations by Strategy")
    for bar, val in zip(bars, shift_rates):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    # Right panel: Clarification rates
    bars = axes[1].bar(strategies_names, clarify_rates, color=colors_bars, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Clarification Rate (%)")
    axes[1].set_title("Clarification Rate by Strategy")
    for bar, val in zip(bars, clarify_rates):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    plt.suptitle("Confidence-Aware Strategy vs Baselines", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_strategy_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_strategy_comparison.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 5: Metric Correlation with Judge (Exp 2) ──
    valid = [r for r in exp2 if r["judge_label"] in ("preserved", "changed")]
    if len(valid) >= 10:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        judge_binary = [1 if r["judge_label"] == "preserved" else 0 for r in valid]

        for ax, (key, label) in zip(axes, [
            ("semantic_similarity", "Semantic Similarity"),
            ("edit_ratio", "Edit Ratio"),
            ("nli_forward", "NLI Forward Entailment"),
        ]):
            vals = [r[key] for r in valid]
            preserved_vals = [v for v, j in zip(vals, judge_binary) if j == 1]
            changed_vals = [v for v, j in zip(vals, judge_binary) if j == 0]

            parts = ax.violinplot([preserved_vals, changed_vals], positions=[0, 1], showmedians=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Preserved\n(Judge)", "Changed\n(Judge)"])
            ax.set_ylabel(label)
            ax.set_title(label)

        plt.suptitle("Automated Metrics vs LLM-as-Judge Labels", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig5_metric_validation.png"), bbox_inches="tight")
        plt.savefig(os.path.join(PLOTS_DIR, "fig5_metric_validation.png"), bbox_inches="tight")
        plt.close()

    # ── Figure 6: By-Dataset Comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset, color in [("banking77", "#2196F3"), ("clinc150", "#FF9800")]:
        rates = []
        labels = []
        for prompt in CORRECTION_PROMPTS.keys():
            subset = [r for r in exp1 if r["dataset"] == dataset and r["prompt_strategy"] == prompt]
            if subset:
                shift_rate = np.mean([1 if r["intent_shifted"] else 0 for r in subset]) * 100
                rates.append(shift_rate)
                labels.append(prompt.replace("_", "\n"))

        ax.plot(labels, rates, "o-", color=color, label=dataset, markersize=8, linewidth=2)

    ax.set_ylabel("Intent Violation Rate (%)")
    ax.set_title("Intent Violation Rates by Dataset and Strategy (Averaged Across Models)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_by_dataset.png"), bbox_inches="tight")
    plt.savefig(os.path.join(PLOTS_DIR, "fig6_by_dataset.png"), bbox_inches="tight")
    plt.close()

    print(f"\n  Saved all figures to {FIGURES_DIR}/ and {PLOTS_DIR}/")


def generate_summary_stats(exp1, exp2, exp3, results_table):
    """Generate summary statistics for the report."""
    summary = {
        "experiment_1": {
            "total_queries": len(exp1),
            "total_intent_shifts": sum(1 for r in exp1 if r["intent_shifted"]),
            "overall_shift_rate": np.mean([1 if r["intent_shifted"] else 0 for r in exp1]),
            "by_model_prompt": {},
            "by_dataset": {},
        },
        "experiment_2": {
            "total_judged": len(exp2),
            "valid_judgments": sum(1 for r in exp2 if r["judge_label"] in ("preserved", "changed")),
        },
        "experiment_3": {
            "confidence_distribution": exp3["confidence_distribution"],
        },
    }

    for (model, prompt), data in results_table.items():
        summary["experiment_1"]["by_model_prompt"][f"{model}/{prompt}"] = {
            "shift_rate": data["shift_rate"],
            "shift_ci": data["shift_ci"],
            "edit_ratio": data["edit_ratio_mean"],
            "semantic_similarity": data["sem_sim_mean"],
        }

    for dataset in ["banking77", "clinc150"]:
        subset = [r for r in exp1 if r["dataset"] == dataset]
        summary["experiment_1"]["by_dataset"][dataset] = {
            "shift_rate": np.mean([1 if r["intent_shifted"] else 0 for r in subset]),
            "n": len(subset),
        }

    with open(os.path.join(DATA_DIR, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main():
    print("Loading results...")
    exp1, exp2, exp3 = load_results()
    print(f"  Exp1: {len(exp1)} results")
    print(f"  Exp2: {len(exp2)} results")
    print(f"  Exp3: {len(exp3['confidence_aware'])} confidence-aware + {len(exp3['always_correct'])} always-correct")

    results_table = analyze_experiment_1(exp1)
    correlations = analyze_experiment_2(exp2)
    strategies = analyze_experiment_3(exp3)

    print("\n\nCreating visualizations...")
    create_visualizations(exp1, exp2, exp3, results_table)

    print("\n\nGenerating summary statistics...")
    summary = generate_summary_stats(exp1, exp2, exp3, results_table)

    print("\n\nDone! Results saved to:")
    print(f"  - {DATA_DIR}/summary_stats.json")
    print(f"  - {FIGURES_DIR}/ (all figures)")
    print(f"  - {PLOTS_DIR}/ (duplicate figures)")


if __name__ == "__main__":
    main()
