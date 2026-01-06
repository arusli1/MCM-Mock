import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------------
# Paths
# -----------------------------
def _default_data_path() -> str:
    # Prefer project-local data file
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "data", "2024_Wimbledon_featured_matches.csv")
    return candidate


# -----------------------------
# 1) Load + basic cleaning
# -----------------------------
def load_data(path: str | None = None) -> pd.DataFrame:
    data_path = path or _default_data_path()
    df = pd.read_csv(data_path)

    # Ensure chronological order within each match
    df = df.sort_values(["match_id", "set_no", "game_no", "point_no"]).reset_index(drop=True)

    # Pre-point progress proxy (optional; not used in the baseline features by default)
    df["point_index_in_match"] = df.groupby("match_id").cumcount()

    # Tiebreak indicator (in this dataset, game_no reaches 13 only for tiebreak)
    df["is_tiebreak"] = (df["game_no"] == 13).astype(int)

    return df


# -----------------------------
# 2) Derived pre-point flags (game/set/match point)
# -----------------------------
def _is_game_point_columns(p1_score: pd.Series, p2_score: pd.Series) -> tuple[pd.Series, pd.Series]:
    s1 = p1_score.astype(str).str.upper()
    s2 = p2_score.astype(str).str.upper()
    # Normal-game "game point" conditions (tiebreak handled separately)
    p1_gp = (s1.eq("AD")) | (s1.eq("40") & s2.isin(["0", "15", "30"]))
    p2_gp = (s2.eq("AD")) | (s2.eq("40") & s1.isin(["0", "15", "30"]))
    return p1_gp.astype(int), p2_gp.astype(int)


def derive_point_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Game points only make sense in non-tiebreak games
    p1_gp, p2_gp = _is_game_point_columns(df["p1_score"], df["p2_score"])
    p1_gp = (p1_gp & (df["is_tiebreak"] == 0)).astype(int)
    p2_gp = (p2_gp & (df["is_tiebreak"] == 0)).astype(int)

    out["is_game_point_p1"] = p1_gp
    out["is_game_point_p2"] = p2_gp

    # Set point (non-tiebreak approximation): if winning this game would close the set
    p1_would_win_set = ((df["p1_games"] + 1 >= 6) & ((df["p1_games"] + 1) - df["p2_games"] >= 2)).astype(int)
    p2_would_win_set = ((df["p2_games"] + 1 >= 6) & ((df["p2_games"] + 1) - df["p1_games"] >= 2)).astype(int)
    out["is_set_point_p1"] = (p1_gp & (df["is_tiebreak"] == 0) & p1_would_win_set).astype(int)
    out["is_set_point_p2"] = (p2_gp & (df["is_tiebreak"] == 0) & p2_would_win_set).astype(int)

    # Match point (best-of-5): a set point while already at 2 sets
    out["is_match_point_p1"] = (out["is_set_point_p1"] & (df["p1_sets"] == 2)).astype(int)
    out["is_match_point_p2"] = (out["is_set_point_p2"] & (df["p2_sets"] == 2)).astype(int)

    return out


# -----------------------------
# 3) Build baseline features (PRE-POINT only)
# -----------------------------
def build_baseline_Xy(df: pd.DataFrame):
    """
    Target y: Player1 wins point? (point_victor == 1)
    
    Baseline features (pre-point context only):
    - Server: dominant factor in tennis (serving player has ~60-70% win rate)
    - Point score: captures leverage within game (40-0 vs 0-40 very different)
    - Set score: captures match-level pressure (late sets more critical)
    
    Rationale:
    - We exclude games score (p1_games, p2_games) because:
      * Highly correlated with point score (if ahead in games, likely ahead in points)
      * Would over-explain match state, leaving less room for "momentum" to show
      * Point score already captures in-game leverage
    - This minimal baseline captures expected performance from rules/context,
      leaving residuals to capture actual performance above/below expectation.
    """
    y = (df["point_victor"] == 1).astype(int).to_numpy()

    # Pre-point context features (minimal but sufficient)
    X = pd.DataFrame({
        # Serve context (dominant pre-point factor)
        "is_p1_serving": (df["server"] == 1).astype(int),

        # Set score (current match) - captures match-level pressure
        "p1_sets": df["p1_sets"].astype(int),
        "p2_sets": df["p2_sets"].astype(int),

        # Point score state (categorical) - captures in-game leverage
        "p1_score": df["p1_score"].astype(str),
        "p2_score": df["p2_score"].astype(str),
    })

    groups = df["match_id"].to_numpy()
    return X, y, groups


# -----------------------------
# 4) One-hot encode + train XGBoost baseline
# -----------------------------
def train_xgb_baseline(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, exclude_match_id: str | None = None):
    """
    Trains XGBoost on match-level split (no leakage across points of same match).
    If exclude_match_id is provided, that match is excluded from training entirely.
    Returns trained booster + encoder so you can transform new points the same way.
    """
    # Exclude target match from training if specified
    if exclude_match_id is not None:
        train_mask = groups != exclude_match_id
        X = X.loc[train_mask].copy()
        y = y[train_mask]
        groups = groups[train_mask]
        print(f"Excluded match '{exclude_match_id}' from training. Training on {len(X)} points from {len(np.unique(groups))} matches.")
    
    # Minimal baseline uses numeric-only features, but keep this generic:
    possible_cat_cols = ["p1_score", "p2_score"]
    cat_cols = [c for c in possible_cat_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    Xtr_num = csr_matrix(X_tr[num_cols].to_numpy())
    Xte_num = csr_matrix(X_te[num_cols].to_numpy())

    enc = None
    if len(cat_cols) > 0:
        # One-hot for categorical score columns (support older/newer sklearn)
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
        Xtr_cat = enc.fit_transform(X_tr[cat_cols])
        Xte_cat = enc.transform(X_te[cat_cols])
        Xtr = hstack([Xtr_num, Xtr_cat], format="csr")
        Xte = hstack([Xte_num, Xte_cat], format="csr")
    else:
        Xtr = Xtr_num
        Xte = Xte_num

    dtr = xgb.DMatrix(Xtr, label=y_tr)
    dte = xgb.DMatrix(Xte, label=y_te)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=2000,
        evals=[(dtr, "train"), (dte, "valid")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    # Holdout logloss (sanity check)
    p_te = booster.predict(dte)
    eps = 1e-12
    logloss = -np.mean(y_te * np.log(p_te + eps) + (1 - y_te) * np.log(1 - p_te + eps))
    print(f"Holdout logloss: {logloss:.4f}")

    return booster, enc, num_cols, cat_cols


def predict_proba(booster, enc, num_cols, cat_cols, X: pd.DataFrame) -> np.ndarray:
    X_num = csr_matrix(X[num_cols].to_numpy())
    if enc is not None and len(cat_cols) > 0:
        X_cat = enc.transform(X[cat_cols])
        Xm = hstack([X_num, X_cat], format="csr")
    else:
        Xm = X_num
    dm = xgb.DMatrix(Xm)
    return booster.predict(dm)


# -----------------------------
# 5) Residuals + EWMA flow curve
# -----------------------------
def ewma(residuals: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Exponential weighted moving average of residuals."""
    F = np.zeros_like(residuals, dtype=float)
    for i, e in enumerate(residuals):
        F[i] = (1 - alpha) * (F[i - 1] if i > 0 else 0.0) + alpha * e
    return F


def ewma_by_set(residuals: np.ndarray, set_no: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    EWMA of residuals that resets at each set boundary.
    This makes performance within each set independent and more interpretable.
    Lower alpha = smoother (less noisy).
    """
    F = np.zeros_like(residuals, dtype=float)
    for i, e in enumerate(residuals):
        # Reset if this is the start of a new set
        if i == 0 or set_no[i] != set_no[i - 1]:
            # Start with the residual itself (not scaled by alpha) for better visibility
            F[i] = e
        else:
            F[i] = (1 - alpha) * F[i - 1] + alpha * e
    return F


def smooth_ewma_by_set(residuals: np.ndarray, set_no: np.ndarray, alpha1: float = 0.12, alpha2: float = 0.10) -> np.ndarray:
    """
    Double-smoothed EWMA (EWMA of EWMA) for less noise.
    Applies two passes of EWMA with resets at set boundaries.
    """
    # First pass
    F1 = ewma_by_set(residuals, set_no, alpha=alpha1)
    # Second pass (smooths the already-smoothed values)
    F2 = ewma_by_set(F1, set_no, alpha=alpha2)
    return F2


def cumulative_by_set(residuals: np.ndarray, set_no: np.ndarray) -> np.ndarray:
    """
    Cumulative sum of residuals within each set (resets at set boundaries).
    Clear indicator: positive = P1 outperformed in this set, negative = P2.
    """
    cumsum = np.zeros_like(residuals, dtype=float)
    current_set = None
    running_sum = 0.0
    for i, (r, s) in enumerate(zip(residuals, set_no)):
        if current_set is None or s != current_set:
            # New set: reset
            current_set = s
            running_sum = r
        else:
            # Same set: accumulate
            running_sum += r
        cumsum[i] = running_sum
    return cumsum


def compute_point_weights(df_match: pd.DataFrame) -> np.ndarray:
    """
    Compute weights for each point based on point "importance" and "quality".
    Higher weight = point matters more for performance evaluation.
    
    Factors considered:
    - Leverage: break points, game points, set points, match points
    - Point quality: longer rallies, more running (when available)
    - Point outcome quality: reduce weight for unearned points (aces, unforced errors)
    
    IMPORTANT: Only applies weighting when data is actually available.
    Missing data = no adjustment (multiplies by 1.0), avoiding bias from imputation.
    
    Returns array of weights (normalized to have mean=1.0).
    """
    weights = np.ones(len(df_match), dtype=float)
    
    # CRITICAL POINTS: These matter most for performance evaluation
    # Break points (highest leverage)
    if "p1_break_pt" in df_match.columns and "p2_break_pt" in df_match.columns:
        is_break_pt = (df_match["p1_break_pt"] == 1) | (df_match["p2_break_pt"] == 1)
        weights[is_break_pt] *= 2.0  # 2x weight for break points
    
    # Game points (important leverage)
    if "is_game_point_p1" in df_match.columns and "is_game_point_p2" in df_match.columns:
        is_game_pt = (df_match["is_game_point_p1"] == 1) | (df_match["is_game_point_p2"] == 1)
        weights[is_game_pt] *= 1.5  # 1.5x for game points
    
    # Set points (very important)
    if "is_set_point_p1" in df_match.columns and "is_set_point_p2" in df_match.columns:
        is_set_pt = (df_match["is_set_point_p1"] == 1) | (df_match["is_set_point_p2"] == 1)
        weights[is_set_pt] *= 2.5  # 2.5x for set points
    
    # Match points (most critical)
    if "is_match_point_p1" in df_match.columns and "is_match_point_p2" in df_match.columns:
        is_match_pt = (df_match["is_match_point_p1"] == 1) | (df_match["is_match_point_p2"] == 1)
        weights[is_match_pt] *= 3.0  # 3x for match points
    
    # Rally length: longer rallies = more skill contest = higher weight
    # Only weight points where rally_count data is actually available
    if "rally_count" in df_match.columns:
        rally = df_match["rally_count"].astype(float)
        has_rally_data = ~rally.isna()
        rally_valid = rally[has_rally_data].clip(lower=1.0)  # Ensure >= 1
        # Log scale: rally of 1→1.0, 5→1.3, 10→1.5, 20→1.7 weight
        rally_adjustment = np.maximum(rally_valid - 1, 0)
        rally_weights = 1.0 + 0.3 * np.log1p(rally_adjustment)
        # Only apply to points with valid data
        weights[has_rally_data] *= rally_weights.values
    
    # Distance run: more running = more effort = higher weight
    # Only weight points where distance data is available for both players
    if "p1_distance_run" in df_match.columns and "p2_distance_run" in df_match.columns:
        dist1 = df_match["p1_distance_run"].astype(float)
        dist2 = df_match["p2_distance_run"].astype(float)
        has_dist_data = ~(dist1.isna() | dist2.isna())
        total_dist = (dist1 + dist2)[has_dist_data]
        total_dist = total_dist.clip(lower=0.0, upper=200.0)  # Clip to reasonable range
        # Scale: 0m→1.0, 50m→1.2, 100m→1.4 weight
        dist_weights = 1.0 + 0.4 * (total_dist / 100.0).clip(0, 1.5)
        # Only apply to points with valid data
        weights[has_dist_data] *= dist_weights.values
    
    # Reduce weight for "easy" points (aces, unforced errors, double faults)
    if "p1_ace" in df_match.columns and "p2_ace" in df_match.columns:
        is_ace = (df_match["p1_ace"] == 1) | (df_match["p2_ace"] == 1)
        weights[is_ace] *= 0.5  # Ace = less skill contest
    
    if "p1_unf_err" in df_match.columns and "p2_unf_err" in df_match.columns:
        is_ue = (df_match["p1_unf_err"] == 1) | (df_match["p2_unf_err"] == 1)
        weights[is_ue] *= 0.7  # Unforced error = point given away
    
    if "p1_double_fault" in df_match.columns and "p2_double_fault" in df_match.columns:
        is_df = (df_match["p1_double_fault"] == 1) | (df_match["p2_double_fault"] == 1)
        weights[is_df] *= 0.6  # Double fault = unearned point
    
    # Handle any NaN or Inf values before normalization
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Normalize so mean weight = 1.0 (keeps residual scale similar)
    mean_weight = weights.mean()
    if mean_weight > 0 and np.isfinite(mean_weight):
        weights = weights / mean_weight
    else:
        # Fallback: all weights = 1.0 if normalization fails
        weights = np.ones_like(weights)
    
    return weights


def weighted_residual(residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply point-quality weights to residuals."""
    return residuals * weights


def rolling_mean(residuals: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling mean of residuals over last N points."""
    return pd.Series(residuals).rolling(window=window, min_periods=1).mean().to_numpy()


# -----------------------------
# 6) Visualization for one match
# -----------------------------
def plot_match_flow(df_match: pd.DataFrame, p: np.ndarray, flow: np.ndarray, title: str):
    # Mark game boundaries for vertical lines
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    x = np.arange(len(df_match))

    plt.figure(figsize=(12, 4))
    plt.plot(x, flow, color="#1f77b4", label="Flow (EWMA residual)")
    plt.axhline(0, color="gray", lw=1, alpha=0.7)
    plt.fill_between(x, 0, flow, where=(flow >= 0), color="#1f77b4", alpha=0.15)
    plt.fill_between(x, 0, flow, where=(flow < 0), color="#d62728", alpha=0.15)
    plt.title(title)
    plt.xlabel("Point index (within match)")
    plt.ylabel("Flow (y − p), EWMA")

    # Light structure markers
    for i in np.where(game_change)[0]:
        plt.axvline(i, color="gray", lw=0.5, alpha=0.25)
    for i in np.where(set_change)[0]:
        plt.axvline(i, color="black", lw=1.0, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_match_performance(df_match: pd.DataFrame, flow: np.ndarray, title: str):
    """
    Performance metric: EWMA of residuals (y - p).
      > 0 means P1 is outperforming baseline expectation
      < 0 means P2 is outperforming baseline expectation
    """
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    x = np.arange(len(df_match))
    plt.figure(figsize=(12, 3.8))
    plt.plot(x, flow, color="#2ca02c", label="Performance (EWMA of y − p)")
    plt.axhline(0, color="gray", lw=1, alpha=0.7)
    plt.fill_between(x, 0, flow, where=(flow >= 0), color="#2ca02c", alpha=0.12)
    plt.fill_between(x, 0, flow, where=(flow < 0), color="#d62728", alpha=0.12)

    # Game / set boundaries
    for i in np.where(game_change)[0]:
        plt.axvline(i, color="gray", lw=0.5, alpha=0.25)
    for i in np.where(set_change)[0]:
        plt.axvline(i, color="black", lw=1.0, alpha=0.35)

    plt.title(title + " — Performance (EWMA residual)")
    plt.xlabel("Point index (within match)")
    plt.ylabel("EWMA(y − p)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_match_summary(df_match: pd.DataFrame, p: np.ndarray, flow: np.ndarray | None, residual: np.ndarray, title: str):
    """
    Clean visualization with baseline + 3 best performance metrics.
    
    Performance metric selection rationale:
    1. Weighted EWMA (reset per set, α=0.20): 
       - Quality-adjusted (longer rallies, break points weighted higher)
       - Responsive (α=0.20 gives ~3.5 point half-life, captures short-term momentum)
       - Resets per set (each set independent, aligns with tennis structure)
       - Best for: identifying who is playing better RIGHT NOW
    
    2. Cumulative Residual (reset per set):
       - Clear set-level performance: positive = P1 outperformed, negative = P2
       - No smoothing bias: directly sums actual vs expected
       - Resets per set: shows who won each set "on merit" vs baseline
       - Best for: answering "who performed better in set X?"
    
    3. Rolling Mean (15 points):
       - Smooth, game-level view (~1 game window)
       - Less noisy than raw residuals, less laggy than long windows
       - No reset: shows sustained trends across sets
       - Best for: identifying medium-term momentum shifts
    """
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()

    # Server switches (directional)
    server = df_match["server"].to_numpy()
    server_switch_idx = np.where(server[1:] != server[:-1])[0] + 1

    x = np.arange(len(df_match))
    set_no = df_match["set_no"].to_numpy()

    # Compute the 3 best performance metrics
    point_weights = compute_point_weights(df_match)
    
    # Diagnostic: report data availability for weighting
    print("\nPoint weighting diagnostics:")
    if "rally_count" in df_match.columns:
        rally_valid = (~df_match["rally_count"].isna()).sum()
        print(f"  Rally count data: {rally_valid}/{len(df_match)} points ({100*rally_valid/len(df_match):.1f}%)")
    if "p1_distance_run" in df_match.columns and "p2_distance_run" in df_match.columns:
        dist_valid = (~(df_match["p1_distance_run"].isna() | df_match["p2_distance_run"].isna())).sum()
        print(f"  Distance data: {dist_valid}/{len(df_match)} points ({100*dist_valid/len(df_match):.1f}%)")
    print(f"  Weight range: [{point_weights.min():.3f}, {point_weights.max():.3f}], mean: {point_weights.mean():.3f}")
    
    weighted_res = weighted_residual(residual, point_weights)
    
    # Metric 1: Weighted EWMA (quality-adjusted, smooth, reset per set)
    # Use double smoothing for less noise
    flow_weighted_ewma = smooth_ewma_by_set(weighted_res, set_no, alpha1=0.12, alpha2=0.10)
    
    # Metric 2: Cumulative residual (clear set-level performance)
    flow_cumulative = cumulative_by_set(residual, set_no)
    
    # Metric 3: Rolling mean (smooth, game-level view)
    flow_rolling = rolling_mean(residual, window=15)

    # Create clean layout: all 4 plots stacked vertically, all same size, wide format
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.3, left=0.06, right=0.97, top=0.96, bottom=0.06,
                         height_ratios=[1, 1, 1, 1])  # All equal height

    # Plot 1: baseline probability
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x, p, color="#9467bd", label="Baseline P(P1 wins point)", lw=2.0)
    ax0.axhline(0.5, color="gray", lw=1.5, alpha=0.6, linestyle="--", label="Equal chance")
    ax0.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax0.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax0.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax0.grid(True, alpha=0.2, linestyle=":")
    ax0.set_ylim([0, 1])
    for i in np.where(game_change)[0]:
        ax0.axvline(i, color="lightgray", lw=0.5, alpha=0.3)
    for i in np.where(set_change)[0]:
        ax0.axvline(i, color="black", lw=1.5, alpha=0.5, linestyle="-")
    for i in server_switch_idx:
        prev_srv = server[i - 1]
        curr_srv = server[i]
        if prev_srv == 1 and curr_srv == 2:
            ax0.axvline(i, color="#1f77b4", lw=1.2, alpha=0.6, linestyle="--")
        elif prev_srv == 2 and curr_srv == 1:
            ax0.axvline(i, color="#ff7f0e", lw=1.2, alpha=0.6, linestyle="--")

    # Plots 2-4: 3 performance metrics (stacked vertically)
    axes_perf = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[3, 0]),
    ]
    
    metrics = [
        (flow_weighted_ewma, "Weighted Momentum (quality-adjusted, double-smoothed EWMA)", "#2ca02c", 
         "Who is playing better NOW?\nImportant points (break/match points) weighted higher."),
        (flow_cumulative, "Set-Level Performance (cumulative residual, reset per set)", "#9467bd",
         "Who outperformed in each set?\nPositive = P1 better, Negative = P2 better."),
        (flow_rolling, "Sustained Trend (rolling mean, 15 points)", "#ff7f0e",
         "Medium-term momentum (~1 game window).\nShows sustained runs across sets."),
    ]

    for ax, (metric_data, label, color, desc) in zip(axes_perf, metrics):
        # Handle NaN/Inf values - replace with 0
        metric_data = np.nan_to_num(metric_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Debug: print metric stats
        print(f"\n{label.split(chr(10))[0]}:")
        print(f"  Min: {metric_data.min():.4f}, Max: {metric_data.max():.4f}, Mean: {metric_data.mean():.4f}, Std: {metric_data.std():.4f}")
        
        ax.plot(x, metric_data, color=color, lw=2.0, alpha=0.9, label=label.split("\n")[0])
        ax.axhline(0, color="gray", lw=1.5, alpha=0.6, linestyle="--")
        ax.fill_between(x, 0, metric_data, where=(metric_data >= 0), color=color, alpha=0.25)
        ax.fill_between(x, 0, metric_data, where=(metric_data < 0), color="#d62728", alpha=0.25)
        ax.set_ylabel("Performance", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        
        # Ensure axis auto-scales properly and includes 0, handle NaN/Inf
        y_min, y_max = float(metric_data.min()), float(metric_data.max())
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            # Fallback if still NaN/Inf
            y_min, y_max = -0.1, 0.1
        
        y_range = y_max - y_min
        if y_range < 0.01 or not np.isfinite(y_range):  # If values are very small or invalid
            y_center = (y_max + y_min) / 2 if np.isfinite((y_max + y_min) / 2) else 0.0
            ax.set_ylim(y_center - 0.1, y_center + 0.1)
        else:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        for i in np.where(set_change)[0]:
            ax.axvline(i, color="black", lw=1.5, alpha=0.5)
        
        # Only add xlabel to bottom plot
        if ax == axes_perf[-1]:
            ax.set_xlabel("Point index", fontsize=10)
        
        # Add text annotation with description
        ax.text(0.02, 0.98, desc, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3))

    # Save figure
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    match_id = str(df_match["match_id"].iloc[0]) if "match_id" in df_match.columns else "match"
    out_path = os.path.join(results_dir, f"{match_id}_summary.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    plt.show()


def plot_match_probability(df_match: pd.DataFrame, p: np.ndarray, title: str):
    # Mark game and set boundaries
    game_change = df_match["game_no"].diff().fillna(0).ne(0).to_numpy()
    set_change = df_match["set_no"].diff().fillna(0).ne(0).to_numpy()
    # Server switches (directional)
    server = df_match["server"].to_numpy()
    server_switch_idx = np.where(server[1:] != server[:-1])[0] + 1  # indices where server changes at point i

    x = np.arange(len(df_match))

    plt.figure(figsize=(12, 3.5))
    plt.plot(x, p, color="#9467bd", label="Baseline P(P1 wins)")
    plt.axhline(0.5, color="gray", lw=1, alpha=0.7)
    for i in np.where(game_change)[0]:
        plt.axvline(i, color="gray", lw=0.5, alpha=0.25)
    for i in np.where(set_change)[0]:
        plt.axvline(i, color="black", lw=1.0, alpha=0.35)
    # Dotted lines for server switches with direction-specific colors
    added_legend_p1_to_p2 = False
    added_legend_p2_to_p1 = False
    for i in server_switch_idx:
        prev_srv = server[i - 1]
        curr_srv = server[i]
        if prev_srv == 1 and curr_srv == 2:
            plt.axvline(
                i, color="#1f77b4", lw=1.0, alpha=0.6, linestyle="--",
                label="Server switch P1→P2" if not added_legend_p1_to_p2 else None
            )
            added_legend_p1_to_p2 = True
        elif prev_srv == 2 and curr_srv == 1:
            plt.axvline(
                i, color="#ff7f0e", lw=1.0, alpha=0.6, linestyle="--",
                label="Server switch P2→P1" if not added_legend_p2_to_p1 else None
            )
            added_legend_p2_to_p1 = True
    plt.title(title + " — Baseline probability")
    plt.xlabel("Point index (within match)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _pick_match_id(df: pd.DataFrame, preferred: str = "2023-wimbledon-1701") -> str:
    if preferred in set(df["match_id"].unique()):
        return preferred
    return df["match_id"].iloc[0]


# -----------------------------
# Run end-to-end
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    
    # Choose target match FIRST (before building X,y)
    target_match_id = "2023-wimbledon-1701"  # Alcaraz vs Djokovic final
    if target_match_id not in set(df["match_id"].unique()):
        target_match_id = df["match_id"].iloc[0]
        print(f"Warning: Target match '2023-wimbledon-1701' not found. Using '{target_match_id}' instead.")
    
    # Build features for all data
    X, y, groups = build_baseline_Xy(df)

    # Train on ALL matches EXCEPT the target match
    print(f"\nTraining baseline model (excluding match '{target_match_id}')...")
    booster, enc, num_cols, cat_cols = train_xgb_baseline(X, y, groups, exclude_match_id=target_match_id)

    # Extract target match for evaluation
    mask = (df["match_id"] == target_match_id)
    df_m = df.loc[mask].copy()
    
    # Add derived flags for point weighting
    flags = derive_point_flags(df_m)
    for col in flags.columns:
        df_m[col] = flags[col]
    
    X_m = X.loc[mask].copy()
    y_m = y[mask]

    print(f"\nEvaluating on match '{target_match_id}' ({len(df_m)} points)...")
    p_m = predict_proba(booster, enc, num_cols, cat_cols, X_m)
    residual_m = y_m - p_m
    
    # Compute match-level performance summary
    match_logloss = -np.mean(y_m * np.log(p_m + 1e-12) + (1 - y_m) * np.log(1 - p_m + 1e-12))
    print(f"Match logloss: {match_logloss:.4f}")
    print(f"Mean absolute residual: {np.abs(residual_m).mean():.4f}")

    title = f"Match Flow — {df_m['player1'].iloc[0]} vs {df_m['player2'].iloc[0]}"
    plot_match_summary(df_m, p_m, None, residual_m, title=title)
