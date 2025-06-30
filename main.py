import os
from deduplicator import detect_duplicates, remove_duplicates, save_to_csv

# Path to dataset
csv_path = "data/Bengaluru_House_Data.csv"

# You can optionally specify columns to deduplicate on
# e.g., subset_columns = ['availability', 'location', 'size']
subset_columns = None  # Set to None for full-row duplicate detection

# Step 1: Detect duplicates
df, duplicates = detect_duplicates(csv_path, subset_columns=subset_columns)

print(f"📊 Total rows in dataset: {len(df)}")
print(f"⚠️ Duplicate rows found: {len(duplicates)}\\n")

if not duplicates.empty:
    print("📄 Sample Duplicates:")
    print(duplicates.head(5).to_markdown(index=False))

# Step 2: Remove duplicates
cleaned = remove_duplicates(df, subset_columns=subset_columns)
print(f"✅ Cleaned dataset has {len(cleaned)} rows\\n")

# Step 3: Save cleaned file
output_path = "data/cleaned_salary.csv"
save_to_csv(cleaned, output_path)
print(f"💾 Cleaned data saved to: {output_path}")