"""22_freeze_beta_and_compare.py -- freezes beta on a validation split and
reports the final uniform-vs-beta* comparison on a separate, untouched test
split, so the reported number isn't inflated by picking whichever beta
happened to look best on the same texts it's then evaluated on.

Pure post-processing of scripts/20's output (results/outputs/20_results.csv)
-- no GPU, no new model calls. Run script 20 with CHENGYU_N_TEXTS set to
however many texts you want split between validation and test, THEN this.

Split: the first CHENGYU_N_VAL texts (by text_id, i.e. the order script 20
already produced them in) are validation; every remaining text is test.
Default N_VAL is half the texts present in the CSV, not hardcoded to any
particular run size."""
import os

import pandas as pd

results = pd.read_csv("results/outputs/20_results.csv")
text_ids = sorted(results["text_id"].unique())
betas = sorted(results["beta"].unique())

N_VAL = int(os.environ.get("CHENGYU_N_VAL", str(len(text_ids) // 2)))
assert 0 < N_VAL < len(text_ids), (
    f"N_VAL={N_VAL} leaves no texts for validation and/or test "
    f"(only {len(text_ids)} texts in the CSV)")

val_ids = set(text_ids[:N_VAL])
test_ids = set(text_ids[N_VAL:])
print(f"{len(text_ids)} texts total: {len(val_ids)} validation, {len(test_ids)} test")

# selection rule, fixed in advance: lowest mean final TVD at the final
# checkpoint, seeds averaged within each text first, then texts averaged --
# same aggregation order as script 20's own cross-text summary
val = results[results["text_id"].isin(val_ids)]
per_text_val = val.groupby(["beta", "text_id"])["final_tvd"].mean()
mean_tvd_by_beta = per_text_val.groupby("beta").mean()
beta_star = mean_tvd_by_beta.idxmin()
print("\nvalidation mean TVD by beta (selection only, not reported as a result):")
for beta in betas:
    marker = "  <- selected" if beta == beta_star else ""
    print(f"  {'uniform' if beta == 0 else f'beta={beta}':>10}: {mean_tvd_by_beta[beta]:.4f}{marker}")

# final report: uniform vs beta_star ONLY, on test texts ONLY -- beta_star
# is already fixed above, nothing here is chosen using these rows
test = results[results["text_id"].isin(test_ids)]
metrics = ["final_tvd", "target_visit_rate", "mode_target_match", "mode_map_match",
           "judge_approves_mode", "last_target_match", "last_map_match",
           "judge_approves_last", "mode_equals_last",
           "acceptance_rate", "movement_rate", "map_hit_ever"]
print(f"\n=== frozen comparison on {len(test_ids)} test text(s): uniform vs beta={beta_star} ===")
for beta in (0, beta_star):
    if beta == 0 and beta_star == 0:
        continue   # uniform selected as beta_star: nothing to compare against itself
    name = "uniform" if beta == 0 else f"beta={beta}"
    sub = test[test["beta"] == beta]
    agg = sub.groupby("text_id")[metrics].mean().mean()
    print(f"  {name:>10}: TVD={agg['final_tvd']:.4f}  visit_rate={agg['target_visit_rate']:.1%}  "
          f"mode-target={agg['mode_target_match']:.1%}  mode-MAP={agg['mode_map_match']:.1%}  "
          f"judge-approves(mode)={agg['judge_approves_mode']:.1%}")
    print(f"  {'':>10}  last-target={agg['last_target_match']:.1%}  "
          f"last-MAP={agg['last_map_match']:.1%}  "
          f"judge-approves(last)={agg['judge_approves_last']:.1%}  "
          f"mode==last={agg['mode_equals_last']:.1%}")
    print(f"  {'':>10}  acceptance={agg['acceptance_rate']:.1%}  "
          f"movement={agg['movement_rate']:.1%}  "
          f"MAP-hit-ever={agg['map_hit_ever']:.1%}")
