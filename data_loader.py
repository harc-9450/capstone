from sklearn.datasets import fetch_20newsgroups
import os

# Load all categories
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Save first N articles into /data folder
os.makedirs("data", exist_ok=True)
for i in range(10):
    with open(f"data/doc_{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(data.data[i])