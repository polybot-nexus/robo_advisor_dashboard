import time

import pandas as pd

df = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined_copy.csv')

new_dataset = df[df['source'] == 'train'].copy()

remaining_rows = df[df['source'] != 'train']

for i, row in remaining_rows.iterrows():
    new_dataset = pd.concat(
        [new_dataset, pd.DataFrame([row])], ignore_index=True
    )
    print(f"Row {i} added to the new dataset.")

    new_dataset.to_csv(
        'datasets/oect_summary_posted_rf__plus_ml_combined.csv', index=False
    )

    time.sleep(30)
