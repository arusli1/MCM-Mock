## Momentum in Tennis — Baseline Flow Model

This repository provides a baseline point-level model that captures the flow of play during a tennis match using only pre-point “context” (server, score state, set/game counts, break-point flags, etc.). It estimates the probability that Player 1 will win the next point, then measures performance relative to that expectation and visualizes the evolving flow (“momentum”) within a match.

### Data
- Input CSV (already included): `data/2024_Wimbledon_featured_matches.csv`
- Data dictionary: `data/2024_data_dictionary.csv`

### Dependencies
- Python 3.10+
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `scipy`

Install (example):
```bash
pip install pandas numpy scikit-learn xgboost matplotlib scipy
```

### How it works

**Baseline Model (XGBoost):**
Predicts `P(Player1 wins current point)` using minimal pre-point context:
- **Server** (`is_p1_serving`): Dominant factor (~60-70% serve win rate)
- **Point score** (`p1_score`, `p2_score`): Captures in-game leverage (40-0 vs 0-40)
- **Set score** (`p1_sets`, `p2_sets`): Captures match-level pressure

**Rationale for minimal baseline:**
- We exclude games score (`p1_games`, `p2_games`) because it's highly correlated with point score and would over-explain match state, leaving less room for "momentum" to show in residuals.
- This baseline captures expected performance from rules/context, leaving residuals to capture actual performance above/below expectation.

**Performance Metrics (3 complementary views):**

1. **Weighted Momentum** (quality-adjusted EWMA, α=0.20, reset per set):
   - Answers: "Who is playing better RIGHT NOW?"
   - Weights longer rallies and break points higher (more skill contest)
   - Responsive (~3.5 point half-life) but resets per set for independence

2. **Set-Level Performance** (cumulative residual, reset per set):
   - Answers: "Who outperformed in each set?"
   - Direct sum: positive = P1 better, negative = P2 better
   - No smoothing bias, clear set-by-set evaluation

3. **Sustained Trend** (rolling mean, 15 points):
   - Answers: "What's the medium-term momentum?"
   - Smooth game-level view (~1 game window)
   - Shows sustained runs across sets

### Running
```bash
python baseline_model.py
```
- The script will train a baseline model with a match-level split (no leakage across the same match).
- It will then visualize the flow for the 2023 Wimbledon Gentlemen’s Final (`match_id = 2023-wimbledon-1701`) if present; otherwise, it picks the first match in the file.

To force a different match, edit the preferred match id in `baseline_model.py` function `_pick_match_id`.

### Interpreting the plots
- Flow plot: area above zero (blue) means Player 1 is performing better than expected given pre-point context; below zero (red) favors Player 2. Vertical thin lines mark game boundaries, darker lines mark set boundaries.
- Baseline probability plot: the model’s pre-point expectation for `P(Player 1 wins)`, showing serve advantage and leverage dynamics across the match.

### Notes and assumptions
- Only pre-point context is used to avoid leakage from point outcomes (fair baseline for “momentum”).
- Tiebreak set-point detection is approximated (since point counters for tiebreaks are not encoded in `p1_score/p2_score`).
- The approach directly accounts for server advantage and critical points via both given and derived flags.
