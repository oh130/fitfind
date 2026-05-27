# Data Pipeline

This directory contains the preprocessing scripts used to build recommendation training data from the raw H&M dataset.

The pipeline starts from three raw CSV files:
- `data/raw/customers.csv`
- `data/raw/articles.csv`
- `data/raw/transactions_train.csv`

It produces the processed files used by search and recommendation experiments.

## Files

- `build_item_master.py`
  - builds the core item master table used for persona scoring and simulation
- `build_customer_purchase_profile.py`
  - builds the customer-level purchase summary table used for persona scoring
- `build_customer_features.py`
  - builds `data/processed/customer_features.csv`
- `build_article_features.py`
  - builds `data/processed/articles_feature.csv`
- `build_article_price_map.py`
  - builds `data/processed/article_price_map.csv` from raw transactions for API price display backfill
- `build_item_features.py`
  - builds item-level popularity and freshness features
- `build_ranking_train_data.py`
  - builds the ranking training dataset with positive and sampled negative rows
- `build_candidate_training_data.py`
  - builds richer user/item/interaction features for Two-Tower candidate training
- `build_sim_users.py`
  - builds the simulated user pool from customer persona ratios
- `build_simulated_events.py`
  - builds the offline simulated search/view/cart/purchase log
- `validate_simulated_events.py`
  - validates the simulated event log and writes a local JSON summary
- `build_event_splits.py`
  - creates time-based train/valid/test event splits from the simulated log
- `run_data_pipeline.py`
  - runs the full pipeline in the correct order

## Recommended Entry Point

If you only have the three raw dataset files and want the full pipeline to run in order, use:

```bash
python data_pipeline/run_data_pipeline.py
```

This also creates `data/processed/article_price_map.csv`, which the API uses for startup-safe price backfill.
The API does not scan `transactions_train.csv` by default because the raw transaction file is large.

## Mode Guide

The scripts support two runtime modes:
- `test`
- `production`

The mode is set at the top of `run_data_pipeline.py`:

```python
# MODE = "production"
MODE = "test"
```

`run_data_pipeline.py` passes the selected mode to:
- `build_item_features.py`
- `build_ranking_train_data.py`
- `build_candidate_training_data.py`

`build_customer_features.py` and `build_article_features.py` always generate their standard processed outputs.

## Output Files

Common outputs:
- `data/processed/item_master.csv`
- `data/processed/customer_purchase_profile.csv`
- `data/processed/customer_features.csv`
- `data/processed/articles_feature.csv`
- `data/processed/article_price_map.csv`

Test mode outputs:
- `data/processed/item_master_test.csv`
- `data/processed/customer_purchase_profile_test.csv`
- `data/processed/user_persona_scores_test.csv`
- `data/processed/item_persona_scores_test.csv`
- `data/processed/sim_users_test.csv`
- `data/processed/simulated_events_test.csv`
- `data/processed/simulated_events_validation_test.json`
- `data/processed/train_events_test.csv`
- `data/processed/valid_events_test.csv`
- `data/processed/test_events_test.csv`
- `data/processed/event_split_summary_test.json`
- `data/processed/item_features_test.csv`
- `data/processed/train_data_test.csv`
- `data/processed/candidate_user_features_test.csv.gz`
- `data/processed/candidate_item_features_test.csv.gz`
- `data/processed/candidate_interactions_test.csv.gz`
- `data/processed/candidate_segment_candidates_test.csv.gz`
- `data/processed/candidate_train_data_test.csv.gz`
- `data/processed/candidate_manifest_test.json`

Production mode outputs:
- `data/processed/item_features.csv`
- `data/processed/user_persona_scores.csv`
- `data/processed/item_persona_scores.csv`
- `data/processed/sim_users.csv`
- `data/processed/simulated_events.csv`
- `data/processed/simulated_events_validation.json`
- `data/processed/train_events.csv`
- `data/processed/valid_events.csv`
- `data/processed/test_events.csv`
- `data/processed/event_split_summary.json`
- `data/processed/train_data_production.csv`
- `data/processed/candidate_user_features.csv.gz`
- `data/processed/candidate_item_features.csv.gz`
- `data/processed/candidate_interactions.csv.gz`
- `data/processed/candidate_segment_candidates.csv.gz`
- `data/processed/candidate_train_data.csv.gz`
- `data/processed/candidate_manifest.json`

## Manual Execution Order

If you want to run each script manually, use this order:

```bash
python data_pipeline/build_customer_features.py
python data_pipeline/build_article_features.py
python data_pipeline/build_article_price_map.py
python data_pipeline/build_item_master.py
python data_pipeline/build_customer_purchase_profile.py
python data_pipeline/build_user_persona_scores.py
python data_pipeline/build_item_persona_scores.py
python data_pipeline/build_sim_users.py
python data_pipeline/build_simulated_events.py
python data_pipeline/validate_simulated_events.py
python data_pipeline/build_event_splits.py
python data_pipeline/build_item_features.py
python data_pipeline/build_ranking_train_data.py
python data_pipeline/build_candidate_training_data.py
```

## Notes

- Run commands from the repository root.
- Keep raw CSV files local only; do not commit them.
- Run `python data_pipeline/build_article_price_map.py` before starting the API if you want broader price coverage without slow startup.
- Set `ENABLE_RAW_TRANSACTION_PRICE_BACKFILL=1` only for one-off local debugging when scanning the full raw transaction file is acceptable.
- `build_item_master.py` creates the item-level canonical table used by persona scoring and simulation.
- `build_customer_purchase_profile.py` creates the customer-level purchase summary table used to derive persona ratios.
- `build_sim_users.py` and `build_simulated_events.py` are offline data-generation scripts; they do not send API requests.
- `build_event_splits.py` performs the required time-based split for the simulated event log.
- `build_ranking_train_data.py` expects `customer_features.csv` and `articles_feature.csv` to exist first.
- The ranking dataset is purchase-based and uses sampled negatives rather than impression logs.
- `build_candidate_training_data.py` creates a positive-interaction dataset with richer aggregate user/item features for candidate retrieval training.

## 2026-05-13 Persona & Simulation Update

The data pipeline was updated so that persona scoring and simulation now align on the same 9-persona schema:
- `trendsetter`
- `practical`
- `value`
- `brand_loyal`
- `impulse`
- `careful`
- `repeat_stable`
- `color_focus`
- `category_focus`

Key changes:
- `build_user_persona_scores.py` was rebalanced to reduce collapse into a few dominant personas and to keep rare personas usable
- `build_item_persona_scores.py` was rebalanced so item-side top personas are less biased toward a narrow subset
- `build_sim_users.py` and `build_simulated_events.py` now operate against the rebalanced persona outputs

Validation summary:
- `test` mode produced valid event logs with all 9 personas present
- `dev` mode produced `200,000` simulated events
- `simulated_events_validation_dev.json` confirmed:
  - `missing_search_query_rows = 0`
  - `missing_item_rows = 0`
  - all 9 personas appeared in the final `active_persona` distribution

Reference docs:
- `docs/data_simulator_work_report_2026-05-13.md`
- `docs/remaining_issues.txt`
- `persona/*.md`
