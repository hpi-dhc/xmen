import pandas as pd
from tqdm import tqdm
from xmen.evaluation import evaluate, entity_linking_error_analysis
import matplotlib
import matplotlib.pyplot as plt


def eval_thresholds(dataset, pred, thresholds, ks, pbar=None):
    """
    Evaluates a set of predictions for different threshold values and top-k metrics.

    Args:
    - dataset (Iterable): Iterable of dictionaries containing the ground truth entities.
    - pred (Iterable): Iterable of dictionaries containing the predicted entities.
    - thresholds (Iterable): Iterable of float values representing the different threshold values to evaluate.
    - ks (Iterable[int]): Iterable of integer values representing the different top-k metrics to evaluate.
    - pbar (Optional[TqdmProgressBar]): A progress bar object to track the evaluation progress. Default is None.

    Returns:
    A dictionary containing the evaluation metrics for each threshold value and top-k metric.
    """
    all_eval = {}

    steps = len(thresholds) * len(ks)

    if not pbar:
        pbar = tqdm(total=steps)
    for threshold in thresholds:
        all_eval[threshold] = {}
        for k in ks:
            eval = evaluate(
                dataset,
                pred,
                allow_multiple_gold_candidates=False,
                top_k_predictions=k,
                threshold=threshold,
            )
            all_eval[threshold][f"f1@{k}"] = eval["strict"]["fscore"]
            all_eval[threshold][f"p@{k}"] = eval["strict"]["precision"]
            all_eval[threshold][f"r@{k}"] = eval["strict"]["recall"]
            pbar.update(1)
    return all_eval


def run_eval(dataset, preds, ks: list, thresholds: list = None):
    """
    Computes evaluation metrics and entity linking error analysis for a set of predictions.

    Args:
    - dataset (Iterable): Iterable of dictionaries containing the ground truth entities.
    - pred (Iterable): Iterable of dictionaries containing the predicted entities.
    - ks (Iterable[int]): Iterable of integer values representing the different top-k metrics to evaluate.
    - thresholds (Iterable): Iterable of float values representing the different threshold values to evaluate. Defaults to None.

    Returns:
    - evals: a list of dictionaries containing evaluation metrics (e.g., precision, recall, F1)
    - error_dfs: a list of error dataframes, one for each prediction, containing entity linking error analysis
    """
    evals = []
    error_dfs = []

    if thresholds:
        steps = len(thresholds) * len(ks) * len(preds)
    else:
        steps = len(preds)

    with tqdm(total=steps) as pbar:
        for pred in preds:
            if thresholds:
                all_eval = eval_thresholds(dataset, pred, thresholds, ks, pbar)
                evals.append(all_eval)

            error_df = entity_linking_error_analysis(dataset, pred)
            error_dfs.append(error_df)
    return evals, error_dfs


def merge_eval_df(linker_names, evals):
    """
    Merges evaluation metrics into a single dataframe.

    Args:
    - linker_names: a list of strings representing the names of the entity linkers
    - evals: a list of dictionaries containing evaluation metrics for each linker and threshold

    Returns:
    - eval_df: a pandas dataframe with columns representing the evaluation metrics and index
      containing a MultiIndex of linker names and threshold values
    """
    eval_df = []
    for e, l in zip(evals, linker_names):
        for t, res in e.items():
            res["threshold"] = t
            res["linker"] = l
            eval_df.append(res)
    eval_df = pd.DataFrame(eval_df)
    eval_df.set_index(["linker", "threshold"], inplace=True)
    return eval_df


def plot_eval_results_at_k(eval_df, k_range, eval_names):
    """
    Plots evaluation results at various values of k for a set of entity linkers.

    Args:
    - eval_df: a pandas dataframe containing evaluation metrics for each entity linker and threshold
    - k_range: a list of integers representing the cutoffs for precision@k and recall@k
    - eval_names: a list of strings representing the names of the entity linkers

    Returns:
    - None
    """
    # matplotlib.rcParams.update({'font.size': 20})

    fig, axs = plt.subplots(1, 3, figsize=(22, 6))

    for j, key in enumerate(["p", "r", "f1"]):
        ax = axs[j]
        cols = [f"{key}@{i}" for i in k_range]
        subset_df = eval_df[cols]
        subset_df.columns = k_range
        subset_df.T.plot(ax=ax, legend=False, style=[".--", "s:", "x-", "x--", "o-"])
        ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
        ax.set_ylim(0.0, 0.8)
        ax.set_xticks(k_range)
        ax_labels = {"p": "Precision", "r": "Recall", "f1": "F1"}
        ax.set_title(ax_labels[key])
        ax.set_xlabel("k")

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        eval_names,
        loc="center",
        ncol=3,
        bbox_to_anchor=(0.512, -0.1, 0.00, 0.0),
    )

    plt.subplots_adjust(wspace=0.20)

    plt.xticks(k_range)
