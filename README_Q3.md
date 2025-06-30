
# 🧹 Q3: Duplicate Detection and Removal from CSV

## ✅ Problem Statement

> Act as a Python developer. Write code to read and print duplicate records from the provided CSV file, remove them from the original file, and save the updated version.

---

## 💡 Objective

This utility:
- Detects exact duplicates or partial duplicates in a `.csv` file
- Displays how many duplicates were found
- Removes duplicates while keeping the first occurrence
- Saves a cleaned version of the file

---

## 📦 Tools Used

- ✅ Python
- ✅ pandas

---

## 🗂️ Folder Structure

```
q3_csv_deduplicator/
├── data/
│   ├── Salary Dataset.csv          # Original file
│   └── cleaned_salary.csv          # Cleaned output
├── src/
│   └── deduplicator.py             # Logic for detection and cleaning
├── main.py                         # Execution script
```

---

## 🧪 Example Output

```
📊 Total rows in dataset: 22770
⚠️ Duplicate rows found: 0

✅ Cleaned dataset has 22770 rows
💾 Cleaned data saved to: data/cleaned_salary.csv
```

---

## 🔧 How to Use

### Step 1: Install pandas
```bash
pip install pandas
```

### Step 2: Run the script
```bash
python main.py
```

---

## ✍️ Customization

To detect duplicates based on specific columns:
Edit `main.py`:

```python
subset_columns = ['company_name', 'job_title', 'salary']
```

---

## ✅ Deliverables

- [x] Detect duplicates from CSV
- [x] Remove duplicates and save cleaned file
- [x] Print stats and preview of duplicates
