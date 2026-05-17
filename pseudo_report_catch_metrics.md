These results are striking. Let me highlight the key findings:

  best1_catch_pct_adot_adj_graduated is the clear winner. This is simply
  the aDOT-adjusted catch% from the best season, with graduated age
  adjustment. Here's how it compares:

  Replacing QBR only (keeping best2_catch_pct_adot_adj)

  ┌───────────────────────────┬───────┬─────┬───────┬──────┬────────┐
  │       Configuration       │ LogLo │ Bri │ >=Eli │ >=St │ >=Star │
  │                           │  ss   │ er  │  te   │  ud  │  ter   │
  ├───────────────────────────┼───────┼─────┼───────┼──────┼────────┤
  │ v11 (current)             │ 2.347 │ 0.5 │ 0.842 │ 0.77 │ 0.833  │
  │                           │       │ 15  │       │ 8    │        │
  ├───────────────────────────┼───────┼─────┼───────┼──────┼────────┤
  │ QBR => best1_catch_pct_ad │ 1.971 │ 0.4 │ 0.861 │ 0.77 │ 0.851  │
  │ ot_adj_graduated          │       │ 99  │       │ 3    │        │
  ├───────────────────────────┼───────┼─────┼───────┼──────┼────────┤
  │ QBR => best1_cpaa_minus_d │ 2.287 │ 0.5 │ 0.852 │ 0.77 │ 0.835  │
  │ rops_graduated            │       │ 21  │       │ 8    │        │
  ├───────────────────────────┼───────┼─────┼───────┼──────┼────────┤
  │ QBR =>                    │ 2.254 │ 0.5 │ 0.838 │ 0.77 │ 0.828  │
  │ best1_clean_catch_rate    │       │ 11  │       │ 5    │        │
  └───────────────────────────┴───────┴─────┴───────┴──────┴────────┘

  QBR => best1_catch_pct_adot_adj_graduated beats v11 on every metric
  except Stud AUC (0.773 vs 0.778, a trivial difference). The LogLoss
  improvement is massive: 1.971 vs 2.347 (-16%). Elite AUC jumps from
  0.842 to 0.861, Starter AUC from 0.833 to 0.851.

  Replacing both QBR + catch% aDOT adj

  ┌────────────────────────┬───────┬──────┬───────┬──────┬─────────┐
  │     Configuration      │ LogLo │ Brie │ >=Eli │ >=St │ >=Start │
  │                        │  ss   │  r   │  te   │  ud  │   er    │
  ├────────────────────────┼───────┼──────┼───────┼──────┼─────────┤
  │ v11 (current)          │ 2.347 │ 0.51 │ 0.842 │ 0.77 │ 0.833   │
  │                        │       │ 5    │       │ 8    │         │
  ├────────────────────────┼───────┼──────┼───────┼──────┼─────────┤
  │ both =>                │       │      │       │      │         │
  │ catch_pct_adot_adj +   │ 1.993 │ 0.48 │ 0.860 │ 0.79 │ 0.843   │
  │ catch_pct_adot_adj_gra │       │ 3    │       │ 1    │         │
  │ duated                 │       │      │       │      │         │
  ├────────────────────────┼───────┼──────┼───────┼──────┼─────────┤
  │ both =>                │       │ 0.51 │       │ 0.79 │         │
  │ cpaa_minus_drops_grad  │ 2.350 │ 3    │ 0.863 │ 7    │ 0.836   │
  │ + 2x_drops             │       │      │       │      │         │
  └────────────────────────┴───────┴──────┴───────┴──────┴─────────┘

  The pair best1_catch_pct_adot_adj + best1_catch_pct_adot_adj_graduated
  dominates v11 across all five metrics.

  What this means

  The cpaa_minus_drops variants you suggested perform well but are beaten
   by a simpler approach: graduated age-adjusted aDOT-adjusted catch%.
  This is the same aDOT adjustment the model already applies to catch%,
  but with the graduated age multiplier that was only applied to YPRR
  before. The logic is sound — a freshman catching 70% aDOT-adjusted is
  more impressive than a senior catching 70%.

  The "minus drops" penalty adds noise without proportionate signal (era
  drift jumps from 0.027 to 0.095 for the graduated variant). The aDOT
  adjustment already accounts for difficulty; the graduated adjustment
  captures age/development trajectory. Adding drop penalties on top
  muddles the signal.

  The interaction effect from the original catch_pct_swap_report.md was
  that best2_catch_pct_adot_adj helped XGBoost despite having negative
  linear residual. In this analysis, keeping best2_catch_pct_adot_adj
  while replacing QBR with best1_catch_pct_adot_adj_graduated preserves
  that XGBoost interaction while adding a genuinely improved linear
  predictor.

  Want me to:
  1. Update the engineering report with these findings?
  2. Run this through the production ensemble (Bayesian + XGBoost
  holdout)?
  3. Explore the graduated catch% more (different age multipliers, best2
  vs best1)?