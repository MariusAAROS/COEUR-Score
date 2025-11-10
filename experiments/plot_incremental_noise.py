import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from io import StringIO
import matplotlib.pyplot as plt
from copy import deepcopy

def load_results(main_folder_path, experiment_type, run_index):
    current_path = os.path.join(main_folder_path, experiment_type)
    folder_path = os.path.join(current_path, f"run_{run_index}.json")

    with open(folder_path, "r") as f:
        data = json.load(f)
    
    results = data["results"]
    
    # Convert back to original data types
    for key, value in results.items():
        if isinstance(value, str) and key != "model_wise_coherence":
            # Convert JSON string back to DataFrame using StringIO
            df = pd.read_json(StringIO(value))
            if 'index' in df.columns:
                df = df.set_index('index')
            results[key] = df
        elif key == "model_wise_coherence":
            # Convert model_wise_coherence back to dict of DataFrames
            for model_name, json_str in value.items():
                df = pd.read_json(StringIO(json_str))
                if 'index' in df.columns:
                    df = df.set_index('index')
                results[key][model_name] = df
        elif isinstance(value, list):
            # Convert list back to numpy array
            results[key] = np.array(value)
    
    return results

def load_results_multi_seed(main_folder_path, experiment_type):
    current_path = os.path.join(main_folder_path, experiment_type)
    #get all run files
    run_files = [f for f in os.listdir(current_path) if f.startswith("run_") and f.endswith(".json")]
    run_ids = [int(f.split("_")[1].split(".")[0]) for f in run_files]

    all_results = []
    for run_id in run_ids:
        results = load_results(main_folder_path, experiment_type, run_id)
        all_results.append(results)

    return all_results

def plot_noise_vs_scores(results, y, model_name=None, noise_ref_level="story"):
    plt.figure(figsize=(8, 6))
    if noise_ref_level == "story":
        noise_type = "actual_story_information_levels"
    elif noise_ref_level == "epic":
        noise_type = "actual_epic_information_levels"
    
    # Plot the actual information level
    plt.plot(results["target_noise_levels"], results[noise_type], 'bo-', linewidth=2, markersize=8, label='Actual Information Level')

    # Select correct data
    if y == "model_wise_coherence":
        if not model_name:
            model_name = list(results[y].keys())[0]
        data_to_plot = results[y][model_name]
    else:
        data_to_plot = results[y]

    # Plot each column
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_to_plot.columns)))
    for i, col in enumerate(data_to_plot.columns):
        plt.plot(results["target_noise_levels"], data_to_plot[col], linestyle='--', linewidth=2, color=colors[i], label=col)

    # Titles
    titles = {
        "bwise_coverage": "Backlog-wise Coverage Scores",
        "ewise_coverage": "Epic-wise Coverage Scores",
        "swise_coverage": "Story-wise Coverage Scores",
        "model_wise_coherence": f"Model-wise Coherence Scores ({model_name})"
    }
    plt.title(titles.get(y, "Target vs Actual Story Noise Levels"))

    plt.xlabel('Target Story Noise Level')
    plt.ylabel('Scores')
    plt.grid(True, alpha=0.3)

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def plot_noise_vs_scores_multi_seed(
    results_list,
    y,
    model_name=None,
    noise_ref_level="story",
    metrics_to_plot=None,  # NEW PARAM: list of sub-columns to plot, or None for all
):
    """
    Plot noise vs score curves averaged over multiple seeds.

    Args:
        results_list: list of dict-like result objects (each one from one random seed)
        y: key to select metric (e.g., "bwise_coverage", "model_wise_coherence", etc.)
        model_name: only used if y == "model_wise_coherence"
        noise_ref_level: "story" or "epic"
        metrics_to_plot: optional list of column names (sub-metrics) to plot;
                         if None, plot all available
    """
    plt.figure(figsize=(10, 6))

    # choose noise column name
    if noise_ref_level == "story":
        noise_type = "actual_story_information_levels"
    elif noise_ref_level == "epic":
        noise_type = "actual_epic_information_levels"
    else:
        raise ValueError("noise_ref_level must be 'story' or 'epic'")

    # pick model name once if needed
    if y == "model_wise_coherence" and model_name is None:
        first = results_list[0]
        if isinstance(first[y], dict):
            model_name = list(first[y].keys())[0]
        else:
            raise ValueError("Unexpected structure for model_wise_coherence in results")

    rows = []

    for seed_idx, results in enumerate(results_list):
        target_noise_levels = list(results["target_noise_levels"])

        # add actual info levels
        actuals = results.get(noise_type, None)
        if actuals is not None:
            for i, t in enumerate(target_noise_levels):
                rows.append({
                    "target_noise_level": float(t),
                    "metric": "Actual Information Level",
                    "score": float(actuals[i]) if i < len(actuals) else np.nan,
                    "seed": seed_idx
                })

        # get metric table/series
        if y == "model_wise_coherence":
            data_to_plot = results[y].get(model_name)
        else:
            data_to_plot = results.get(y)

        if data_to_plot is None:
            continue

        # normalize to DataFrame
        if isinstance(data_to_plot, pd.Series):
            data_to_plot = data_to_plot.to_frame(name=data_to_plot.name or "value")

        # filter submetrics if requested
        if metrics_to_plot is not None:
            available = [m for m in metrics_to_plot if m in data_to_plot.columns]
            data_to_plot = data_to_plot[available]

        for col in data_to_plot.columns:
            for i, t in enumerate(target_noise_levels):
                val = (
                    data_to_plot.iloc[i, data_to_plot.columns.get_loc(col)]
                    if i < len(data_to_plot)
                    else np.nan
                )
                rows.append({
                    "target_noise_level": float(t),
                    "metric": str(col),
                    "score": float(val) if not pd.isna(val) else np.nan,
                    "seed": seed_idx
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data collected — check results_list or your 'y' key.")

    # aggregate mean and std
    agg = df.groupby(["metric", "target_noise_level"])["score"].agg(["mean", "std"]).reset_index()

    # plot each metric with shaded std band
    for metric in agg["metric"].unique():
        sub = agg[agg["metric"] == metric].sort_values("target_noise_level")
        x, y_mean, y_std = sub["target_noise_level"], sub["mean"], sub["std"].fillna(0)

        if metric == "Actual Information Level":
            plt.plot(x, y_mean, label=metric, linestyle="-", marker="o", linewidth=2.5)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)
        else:
            plt.plot(x, y_mean, label=metric, linestyle="--", linewidth=1.8)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.08)

    # titles, labels, etc.
    titles = {
        "bwise_coverage": "Backlog-wise Coverage Scores (Multi-seed)",
        "ewise_coverage": "Epic-wise Coverage Scores (Multi-seed)",
        "swise_coverage": "Story-wise Coverage Scores (Multi-seed)",
        "model_wise_coherence": f"Model-wise Coherence Scores ({model_name}) (Multi-seed)"
    }
    plt.title(titles.get(y, "Target vs Actual Story Noise Levels (Multi-seed)"))
    plt.xlabel("Target Story Noise Level")
    plt.ylabel("Scores")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def compute_pearson(noise_levels, scores):
    if len(noise_levels) < 25:
        print("Warning: Pearson correlation may not be reliable with less than 25 data points.")
    pearson_results = {}
    for col in scores.columns:
        r, p_value = pearsonr(noise_levels, scores[col].values)
        pearson_results[col] = {"pearson_r": r, "p_value": p_value}
    return pd.DataFrame(pearson_results)

def compute_spearman(noise_levels, scores):
    if len(noise_levels) < 20:
        print("Warning: Spearman correlation may not be reliable with less than 20 data points.")
    spearman_results = {}
    for col in scores.columns:
        r, p_value = spearmanr(noise_levels, scores[col].values)
        spearman_results[col] = {"spearman_r": r, "p_value": p_value}
    return pd.DataFrame(spearman_results)

def compute_corrs_multi_seed(results_list, y, model_name=None, noise_ref_level="story",
                               metrics_to_plot=None):
    """
    Compute Pearson correlation coefficients between actual noise levels
    and metric scores across multiple seeds.
    Returns a DataFrame with mean and std for each metric.
    """
    all_pearson = []

    for results in results_list:
        # select noise type
        if noise_ref_level == "story":
            noise_type = "actual_story_information_levels"
        elif noise_ref_level == "epic":
            noise_type = "actual_epic_information_levels"
        else:
            raise ValueError("noise_ref_level must be 'story' or 'epic'")

        noise_levels = results.get(noise_type)
        if noise_levels is None:
            continue

        # choose metric data
        if y == "model_wise_coherence":
            if model_name is None:
                model_name = list(results[y].keys())[0]
            data_to_plot = results[y].get(model_name)
        else:
            data_to_plot = results.get(y)

        if data_to_plot is None:
            continue

        if metrics_to_plot is not None:
            available = [m for m in metrics_to_plot if m in data_to_plot.columns]
            data_to_plot = data_to_plot[available]

        # normalize to DataFrame
        if isinstance(data_to_plot, pd.Series):
            data_to_plot = data_to_plot.to_frame(name=data_to_plot.name or "value")

        # Compute Pearson correlations for each column
        df_seed_pearson = compute_pearson(noise_levels, data_to_plot)
        df_seed_pearson.rename(index={"p_value": "pearson_p_value"}, inplace=True)
        df_seed_spearman = compute_spearman(noise_levels, data_to_plot)
        df_seed_spearman.rename(index={"p_value": "spearman_p_value"}, inplace=True)
        df_seed = pd.concat([df_seed_pearson, df_seed_spearman], axis=0)
        all_pearson.append(df_seed)

    # Combine results
    if not all_pearson:
        raise RuntimeError("No Pearson correlations computed; check your input structure.")

    combined = pd.concat(all_pearson, ignore_index=False)

    pearson = combined[combined.index == "pearson_r"].agg(["mean", "std"])
    spearmanr = combined[combined.index == "spearman_r"].agg(["mean", "std"])
    pearson_pvalues = combined[combined.index == "pearson_p_value"].agg(["mean", "std"])
    spearman_pvalues = combined[combined.index == "spearman_p_value"].agg(["mean", "std"])

    return pearson, spearmanr, pearson_pvalues, spearman_pvalues

def plot_corrs_multi_seed(corrs_summary, alpha=0.05):
    """
    Plot mean ± std Pearson and Spearman correlations for multiple metrics.
    Expects output from compute_corrs_multi_seed.
    """
    pearson, spearman, pearson_pvalues, spearman_pvalues = corrs_summary
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pearson subplot
    metrics = pearson.columns
    means_p = pearson.loc["mean"]
    stds_p = pearson.loc["std"]
    x = np.arange(len(metrics))
    colors_p = []
    for metric in metrics:
        mean_p_value = pearson_pvalues.loc["mean"][metric]
        if mean_p_value < alpha:
            colors_p.append("green")
        else:
            colors_p.append("red")
    
    ax1.bar(x, means_p, yerr=stds_p, color=colors_p, alpha=0.7, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha="right")
    ax1.set_ylabel("Pearson Correlation Coefficient (r)")
    ax1.set_title("Mean ± Std Pearson Correlation Coefficients")
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.grid(True, alpha=0.3)
    
    # Spearman subplot
    means_s = spearman.loc["mean"]
    stds_s = spearman.loc["std"]
    colors_s = []
    for metric in metrics:
        mean_p_value = spearman_pvalues.loc["mean"][metric]
        if mean_p_value < alpha:
            colors_s.append("green")
        else:
            colors_s.append("red")
    
    ax2.bar(x, means_s, yerr=stds_s, color=colors_s, alpha=0.7, capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45, ha="right")
    ax2.set_ylabel("Spearman Correlation Coefficient (ρ)")
    ax2.set_title("Mean ± Std Spearman Correlation Coefficients")
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def split_coeur_results(results, coherence=True, coverage=True,
                        bwise_coverage=True, ewise_coverage=True, swise_coverage=True,
                        coherence_subcols="default", coverage_subcols="F1", include_mauve=False,
                        coverage_metric=None):
    if coherence:
        if isinstance(coherence_subcols, str) and coherence_subcols == "all":
            small_concat_coherence_scores = results["coherence_scores"].drop(columns=["Total Points", "Number of Clusters", "Noise Points", "noise_level"])
        elif isinstance(coherence_subcols, list):
            small_concat_coherence_scores = results["coherence_scores"][coherence_subcols]
        elif isinstance(coherence_subcols, str) and coherence_subcols == "default":
            small_concat_coherence_scores = results["coherence_scores"][
                ["Adjusted Rand Index", "Normalized Mutual Info",
                "V-Measure", "Fowlkes-Mallows"]]
        model_wise_coherence = {}
        for coherence_index in range(small_concat_coherence_scores.shape[0]):
            model_wise_coherence[small_concat_coherence_scores.index[coherence_index]] = small_concat_coherence_scores[
            small_concat_coherence_scores.index == small_concat_coherence_scores.index[coherence_index]]
        
        results["model_wise_coherence"] = model_wise_coherence

    if coverage:
        if not coverage_metric:
            metrics = ["BERTScore", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        elif isinstance(coverage_metric, list):
            metrics = coverage_metric
        if isinstance(coverage_subcols, str) and coverage_subcols == "F1":
            coverage_cols = [m+" F1" for m in metrics]
        elif isinstance(coverage_subcols, str) and coverage_subcols == "P":
            coverage_cols = [m+" Precision" for m in metrics]
        elif isinstance(coverage_subcols, str) and coverage_subcols == "R":
            coverage_cols = [m+" Recall" for m in metrics]
        elif isinstance(coverage_subcols, str) and coverage_subcols == "all":
            coverage_cols = []
            for m in metrics:
                coverage_cols.extend([m+" Precision", m+" Recall", m+" F1"])
        if include_mauve:
            coverage_cols.extend(["MAUVE", "MAUVE (Star)"])
        small_concat_coverage_scores = results["coverage_scores"][coverage_cols]
        if bwise_coverage:
            bwise_coverage = small_concat_coverage_scores[small_concat_coverage_scores.index == "Backlog-wise"]
            results["bwise_coverage"] = bwise_coverage
        if ewise_coverage:
            ewise_coverage = small_concat_coverage_scores[small_concat_coverage_scores.index.str.contains("Epic-wise")]
            results["ewise_coverage"] = ewise_coverage
        if swise_coverage:
            swise_coverage = small_concat_coverage_scores[small_concat_coverage_scores.index.str.contains("Story-wise")]
            results["swise_coverage"] = swise_coverage
    return results

def save_results(results, main_folder_path, experiment_type):
    results = deepcopy(results)
    current_path = os.path.join(main_folder_path, experiment_type)
    folder_index = len([f for f in os.listdir(current_path) if f.startswith("run_")]) + 1
    folder_path = os.path.join(current_path, f"run_{folder_index}.json")

    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value = value.reset_index()
            results[key] = value.to_json(indent=4)
        elif key == "model_wise_coherence":
            for model_name, df in value.items():
                results[key][model_name] = df.reset_index().to_json(indent=4)
        elif isinstance(value, np.ndarray):
            results[key] = value.tolist()
    with open(folder_path, "w") as f:
        json.dump({"results": results}, f, indent=4)