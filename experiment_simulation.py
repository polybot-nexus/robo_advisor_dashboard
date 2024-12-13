import time

import pandas as pd

df = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined_copy.csv')

# New dataset to which rows will be added
new_dataset = df[df['source'] == 'train'].copy()

# Filter the rows that are not in the "train" source
remaining_rows = df[df['source'] != 'train']

# Simulating adding a new line every minute (Here we reduce time for testing)
for i, row in remaining_rows.iterrows():
    new_dataset = pd.concat(
        [new_dataset, pd.DataFrame([row])], ignore_index=True
    )
    print(f"Row {i} added to the new dataset.")

    # Save the updated new_dataset to CSV every time a row is added
    new_dataset.to_csv(
        'datasets/oect_summary_posted_rf__plus_ml_combined.csv', index=False
    )

    time.sleep(10)
