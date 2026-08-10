import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

def plot_features_dual_axis(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str,
    n_bins: int = 10,
    n_wide: int = 3,
    figsize_per_plot: tuple[float, float] = (5, 4),
    int_as_cat_unique_max: int | None = 20,
    cat_order: dict[str, list] | None = None,   # <-- NEW
):
    BAR_COLOR = "tab:blue"
    LINE_COLOR = "tab:orange"

    def is_categorical(s: pd.Series) -> bool:
        return s.dtype == "object" or pd.api.types.is_string_dtype(s)

    n_cols = len(cols)
    n_rows = int(np.ceil(n_cols / n_wide))

    fig, axes = plt.subplots(
        n_rows,
        n_wide,
        figsize=(figsize_per_plot[0] * n_wide, figsize_per_plot[1] * n_rows),
    )

    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(cols):
        ax1 = axes[i]
        col_safe = col.replace("_", r"\_")

        n_nan = df[col].isna().sum()
        n_unique = df[col].nunique(dropna=True)

        tmp = df[[col, target_col]].dropna()
        if tmp.empty:
            ax1.set_title(
                rf"$\bf{{{col_safe}}}$"
                f"\n(empty after dropna; {n_unique} unique, {n_nan} nan)"
            )
            continue

        x = tmp[col]
        y = tmp[target_col]

        # ---------- TRUE CATEGORICAL ----------
        if is_categorical(x):
            type_str = "categorical"

            # base counts / means
            counts = x.value_counts()
            mean_y = tmp.groupby(col)[target_col].mean()

            # ---- APPLY USER-DEFINED CATEGORY ORDER (if provided) ----
            if cat_order is not None and col in cat_order:
                desired = list(cat_order[col])

                # keep only categories present in data
                ordered = [c for c in desired if c in counts.index]

                # append any remaining categories not specified by user
                remaining = [c for c in counts.index if c not in ordered]
                final_order = ordered + remaining
            else:
                final_order = sorted(counts.index)

            counts = counts.loc[final_order]
            mean_y = mean_y.loc[final_order]

            xpos = np.arange(len(final_order))

            ax1.bar(xpos, counts.values, alpha=0.6, color=BAR_COLOR)
            ax1.set_xlabel(col)
            ax1.set_ylabel("Count", color=BAR_COLOR)
            ax1.tick_params(axis="y", colors=BAR_COLOR)

            ax1.set_xticks(xpos)
            ax1.set_xticklabels(final_order, rotation=45, ha="right")

            ax2 = ax1.twinx()
            ax2.plot(xpos, mean_y.values, marker="o", color=LINE_COLOR)
            ax2.set_ylabel(f"Mean {target_col}", color=LINE_COLOR)
            ax2.tick_params(axis="y", colors=LINE_COLOR)

            ax1.set_title(
                rf"$\bf{{{col_safe}}}$: Count vs Mean {target_col}"
                f"\n({type_str} with {n_unique} unique and {n_nan} nan)"
            )

        # ---------- NUMERIC ----------
        else:
            type_str = "numeric"

            xvals = x.values
            yvals = y.values

            mask = np.isfinite(xvals) & np.isfinite(yvals)
            xvals = xvals[mask]
            yvals = yvals[mask]

            if len(xvals) == 0:
                ax1.set_title(
                    rf"$\bf{{{col_safe}}}$"
                    f"\n({type_str} with {n_unique} unique and {n_nan} nan)"
                )
                continue

            unique_vals = np.sort(np.unique(xvals))
            n_unique_eff = len(unique_vals)

            # --- Case 1: int-as-categorical ---
            if int_as_cat_unique_max is not None and n_unique_eff <= int_as_cat_unique_max:
                counts = np.array([(xvals == v).sum() for v in unique_vals])
                mean_y = np.array([yvals[xvals == v].mean() for v in unique_vals])

                xpos = np.arange(n_unique_eff)

                ax1.bar(xpos, counts, alpha=0.6, color=BAR_COLOR)
                ax1.set_xlabel(col)
                ax1.set_ylabel("Count", color=BAR_COLOR)
                ax1.tick_params(axis="y", colors=BAR_COLOR)

                rotate = 45 if n_unique_eff > n_bins else 0
                step = max(int(np.ceil(n_unique_eff / n_bins)), 1)
                tick_idx = np.arange(0, n_unique_eff, step)

                if pd.api.types.is_integer_dtype(x):
                    tick_labels = unique_vals[tick_idx].astype(int)
                else:
                    tick_labels = unique_vals[tick_idx]

                ax1.set_xticks(tick_idx)
                ax1.set_xticklabels(
                    tick_labels,
                    rotation=rotate,
                    ha="right" if rotate else "center",
                )

                ax2 = ax1.twinx()
                ax2.plot(xpos, mean_y, marker="o", color=LINE_COLOR)
                ax2.set_ylabel(f"Mean {target_col}", color=LINE_COLOR)
                ax2.tick_params(axis="y", colors=LINE_COLOR)

                ax1.set_title(
                    rf"$\bf{{{col_safe}}}$: Per-Value Count vs Mean {target_col}"
                    f"\n({type_str} with {n_unique} unique and {n_nan} nan)"
                )

            # --- Case 2: low-cardinality numeric bins ---
            elif n_unique_eff < n_bins:
                counts = np.array([(xvals == v).sum() for v in unique_vals])
                mean_y = np.array([yvals[xvals == v].mean() for v in unique_vals])

                width = 0.8 * (np.min(np.diff(unique_vals)) if n_unique_eff > 1 else 1.0)

                ax1.bar(unique_vals, counts, width=width, alpha=0.6, color=BAR_COLOR)
                ax1.set_xlabel(col)
                ax1.set_ylabel("Count", color=BAR_COLOR)
                ax1.tick_params(axis="y", colors=BAR_COLOR)

                ax2 = ax1.twinx()
                ax2.plot(unique_vals, mean_y, marker="o", color=LINE_COLOR)
                ax2.set_ylabel(f"Mean {target_col}", color=LINE_COLOR)
                ax2.tick_params(axis="y", colors=LINE_COLOR)

                ax1.set_title(
                    rf"$\bf{{{col_safe}}}$: Per-Value Count vs Mean {target_col}"
                    f"\n({type_str} with {n_unique} unique and {n_nan} nan)"
                )

            # --- Case 3: regular histogram ---
            else:
                bins = np.linspace(xvals.min(), xvals.max(), n_bins + 1)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])

                counts, _ = np.histogram(xvals, bins=bins)
                bin_idx = np.digitize(xvals, bins) - 1

                mean_y = np.array([
                    yvals[bin_idx == j].mean() if np.any(bin_idx == j) else np.nan
                    for j in range(n_bins)
                ])

                ax1.bar(
                    bin_centers,
                    counts,
                    width=(bins[1] - bins[0]),
                    alpha=0.6,
                    color=BAR_COLOR,
                )
                ax1.set_xlabel(col)
                ax1.set_ylabel("Count", color=BAR_COLOR)
                ax1.tick_params(axis="y", colors=BAR_COLOR)

                ax2 = ax1.twinx()
                ax2.plot(bin_centers, mean_y, marker="o", color=LINE_COLOR)
                ax2.set_ylabel(f"Mean {target_col}", color=LINE_COLOR)
                ax2.tick_params(axis="y", colors=LINE_COLOR)

                ax1.set_title(
                    rf"$\bf{{{col_safe}}}$: Histogram vs Mean {target_col}"
                    f"\n({type_str} with {n_unique} unique and {n_nan} nan)"
                )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_train_test_distributions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    n_bins: int = 20,
):
    """
    Plot side-by-side Train vs Test distributions for given columns.

    - Categorical columns: bar plots of value counts
    - Numeric columns: histograms with shared binning
    """

    BAR_ALPHA = 0.6

    for c in cols:
        if c not in train.columns or c not in test.columns:
            print(f"[skip] column '{c}' not found in both train and test")
            continue

        is_cat = is_object_dtype(train[c]) or is_string_dtype(train[c])

        # slightly taller for rotated labels
        if is_cat:
            fig, axes = plt.subplots(1, 2, figsize=(8, 2.4))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8, 2))

        if is_cat:
            # ---------- CATEGORICAL ----------
            cats = sorted(
                set(train[c].dropna().unique())
                | set(test[c].dropna().unique())
            )

            train_counts = train[c].value_counts().reindex(cats, fill_value=0)
            test_counts  = test[c].value_counts().reindex(cats, fill_value=0)

            axes[0].bar(cats, train_counts, alpha=BAR_ALPHA)
            axes[1].bar(cats, test_counts,  alpha=BAR_ALPHA)

            for ax in axes:
                ax.tick_params(axis="x", labelsize=8)
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")

        else:
            # ---------- NUMERIC ----------
            combined = pd.concat([train[c], test[c]]).dropna()
            xmin, xmax = combined.min(), combined.max()

            if xmin == xmax:
                bins = 1
            else:
                bins = np.linspace(xmin, xmax, n_bins + 1)

            axes[0].hist(train[c], bins=bins, alpha=BAR_ALPHA)
            axes[1].hist(test[c],  bins=bins, alpha=BAR_ALPHA)

            for ax in axes:
                ax.set_xlim(xmin, xmax)

        axes[0].set_title(f"Train: {c}")
        axes[1].set_title(f"Test: {c}")

        plt.tight_layout()
        plt.show()

import numpy as np
from typing import Callable, List, Tuple
from collections import defaultdict


# -------------------------------
# Metrics
# -------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# -------------------------------
# Helper functions
# -------------------------------
def normalize_weights(w: np.ndarray) -> np.ndarray:
    s = w.sum()
    if s == 0:
        return w
    return w / s


def ensemble_prediction(preds: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(preds[0], dtype=float)
    for p, w in zip(preds, weights):
        out += w * p
    return out


def optimal_alpha(
    y_true: np.ndarray,
    current_pred: np.ndarray,
    new_pred: np.ndarray,
    metric: Callable,
    alphas: np.ndarray,
) -> Tuple[float, float]:

    best_alpha = 0.0
    best_score = metric(y_true, current_pred)

    for a in alphas:
        blended = (1 - a) * current_pred + a * new_pred
        score = metric(y_true, blended)

        if score < best_score:
            best_score = score
            best_alpha = a

    return best_alpha, best_score


# -------------------------------
# Hill Climbing Ensemble
# -------------------------------
def hill_climb_ensemble(
    oofs: List[np.ndarray],
    test_preds: List[np.ndarray],
    y_true: np.ndarray,
    metric: Callable = rmse,
    max_number_models: int | None = None,
    tolerance: float = 1e-6,
    use_negative_weights: bool = False,
    alpha_grid: np.ndarray | None = None,
    names: List[str] | None = None,
    verbose: bool = True,
):

    if names is None:
        names = [f"model_{i}" for i in range(len(oofs))]
    else:
        assert len(names) == len(oofs)

    if alpha_grid is None:
        alpha_grid = (
            np.linspace(-1, 1, 201)
            if use_negative_weights
            else np.linspace(0, 1, 101)
        )

    n_models = len(oofs)

    # -------------------------------
    # Step 1: best single model
    # -------------------------------
    scores = [metric(y_true, oof) for oof in oofs]
    best_idx = int(np.argmin(scores))

    used_models = [best_idx]
    weights = np.array([1.0])
    current_oof = oofs[best_idx].copy()
    current_score = scores[best_idx]

    if verbose:
        print(f"Initial model: {names[best_idx]}  score={current_score:.6f}")

    # -------------------------------
    # Hill climbing loop
    # -------------------------------
    while True:

        if max_number_models is not None and len(used_models) >= max_number_models:
            if verbose:
                print("Reached max_number_models")
            break

        best_candidate = None
        best_alpha = None
        best_score = current_score

        # current aggregated weights per model
        agg = defaultdict(float)
        for i, w in zip(used_models, weights):
            agg[i] += w

        for i in range(n_models):

            # ----------------------------------------
            # Allow model-specific negative alpha
            # ----------------------------------------
            if use_negative_weights:
                alphas_i = alpha_grid
            else:
                current_w = agg.get(i, 0.0)
                if current_w > 0:
                    # allow removing up to current weight
                    neg = alpha_grid[alpha_grid < 0]
                    neg = neg[neg >= -current_w]
                    pos = alpha_grid[alpha_grid >= 0]
                    alphas_i = np.concatenate([neg, pos])
                else:
                    alphas_i = alpha_grid[alpha_grid >= 0]

                if len(alphas_i) == 0:
                    continue

            alpha, score = optimal_alpha(
                y_true=y_true,
                current_pred=current_oof,
                new_pred=oofs[i],
                metric=metric,
                alphas=alphas_i,
            )

            if score < best_score:
                best_score = score
                best_candidate = i
                best_alpha = alpha

        improvement = current_score - best_score

        if best_candidate is None or improvement < tolerance:
            if verbose:
                print(f"Stopping: improvement {improvement:.3e} < tolerance")
            break

        # Apply update
        weights = (1 - best_alpha) * weights
        weights = np.append(weights, best_alpha)
        weights = normalize_weights(weights)

        used_models.append(best_candidate)
        current_oof = ensemble_prediction([oofs[i] for i in used_models], weights)
        current_score = best_score

        if verbose:
            print(
                f"Add model: {names[best_candidate]}  "
                f"alpha={best_alpha:.4f}  score={current_score:.6f}"
            )

    # -------------------------------
    # Aggregate duplicate models
    # -------------------------------
    agg = defaultdict(float)
    for i, w in zip(used_models, weights):
        agg[i] += w

    final_models = list(agg.keys())
    final_weights = normalize_weights(np.array([agg[i] for i in final_models]))

    final_oof = ensemble_prediction([oofs[i] for i in final_models], final_weights)
    final_test = ensemble_prediction([test_preds[i] for i in final_models], final_weights)

    if verbose:
        print("\nFinal ensemble (collapsed):")
        for i, w in zip(final_models, final_weights):
            print(f"  {names[i]:>20s}  weight={w:.6f}")
        print(f"Final score = {metric(y_true, final_oof):.6f}")

    return {
        "used_models_raw": used_models,
        "used_models": final_models,
        "used_names": [names[i] for i in final_models],
        "weights": final_weights,
        "oof_pred": final_oof,
        "test_pred": final_test,
        "score": metric(y_true, final_oof),
    }