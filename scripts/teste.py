import pandas as pd

df = pd.read_csv(r"C:\Users\Polyana\Documents\pesquisa-P2SQL\data\processed\04_prompt_dataset_final.csv")
print(df["target_table"].value_counts(dropna=False).head(50))

# python scripts/01_prepare_sql_annotation_sheet.py --dataset1 data/raw/dataset1.csv --dataset2 data/raw/dataset2.csv --dataset1-sql-col sql --dataset1-label-col label --dataset2-sql-col sql --dataset2-label-col label --output-dir data/interim

# python scripts/02_consolidate_canonical_intents.py --annotation-sheet data/interim/01_annotation_sheet.csv --output-dir data/processed

# python scripts/03_generate_prompts.py --canonical-intents data/processed/02_canonical_intents.csv --output-dir data/processed --batch-size 2 --max-retries 3

# python scripts/03b_validate_prompt_candidates.py --input data/processed/03_prompt_candidates.csv --output data/processed/03b_prompt_candidates_validated.csv --summary-output data/processed/03b_prompt_candidates_summary.csv

# python scripts/04_refine_prompt_dataset.py --input data/processed/03b_prompt_candidates_validated.csv --output-final data/processed/04_prompt_dataset_final.csv --output-review data/processed/04_prompt_dataset_review_flags.csv --output-rewrite-log data/processed/04_prompt_dataset_rewrite_log.csv --summary-output data/processed/04_prompt_dataset_summary.csv --rewrite-with-llm

# python scripts/05_build_publication_ready_dataset.py --input data/processed/04_prompt_dataset_final.csv --output data/processed/05_prompt_dataset_publication_ready.csv --summary-output data/processed/05_prompt_dataset_publication_ready_summary.csv