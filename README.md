# Q18 – Salary Prediction (Multivariate Linear Regression + PCA)

### 🎯 Objective
Build a multivariate linear regression model to predict salaries based on various attributes including job title, location, employment status, and company type. Explore the impact of dimensionality reduction using PCA.

---

### 📦 Dataset Overview
- **Rows**: 22,770
- **Features**: Job title, company, employment type, rating, location, and salary
- **Target**: Salary (in INR)

---

### ✅ Project Pipeline

1. **Data Preprocessing**
   - Normalized column names
   - Dropped redundant fields (e.g., raw job title where job roles already exist)
   - Bucketed `company_name` into Top 50 + "Other"

2. **Feature Engineering**
   - One-hot encoded categorical features: job roles, company, location, employment type
   - Scaled numerical features using `StandardScaler`

3. **Full Feature Regression**
   - Trained LinearRegression on all 74 features
   - Visualized actual vs predicted salaries and residuals

4. **PCA Dimensionality Reduction**
   - Applied PCA with 2 and 3 components
   - Explained Variance:
     - PCA(2): 5.6%
     - PCA(3): 7.8%
   - Extracted top contributing features for each component

5. **Regression on PCA-reduced Features**
   - Trained new models on PCA(2) and PCA(3)
   - Visualized predictions and residuals

---

### 📊 Final Comparison – Full vs PCA-Based Models

| Model                  | RMSE (₹)     | R² Score | Explained Variance |
|------------------------|-------------|----------|---------------------|
| **Full Model**         | 600,677.75  | 0.110    | —                   |
| **PCA (2 components)** | 635,592.81  | 0.004    | 5.6%                |
| **PCA (3 components)** | 618,520.15  | 0.057    | 7.8%                |

---

### 🔍 PCA Feature Insights

| Component | Dominant Features                             |
|-----------|------------------------------------------------|
| PC1       | Intern vs Full-Time, Company Tier, Java roles |
| PC2       | SDE, IOS roles, Full-Time Status              |
| PC3       | Java vs SDE, Location (e.g., Chennai)         |

---

### 📈 Visualizations
- Actual vs Predicted Salaries (All Models)
- Residual Distributions
- PCA 2D and 3D Projections

---

### 🧠 Conclusion
- Full model outperformed PCA-reduced models, as PCA discarded too much salary-related variance
- PCA revealed core trends in employment and job-type impact on salary
- Company tier, employment status, and role were key drivers of variance

---

### 🛠️ Tech Stack
- Python 3.9
- scikit-learn, pandas, matplotlib, seaborn
