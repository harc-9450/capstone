# Q17 – Bangalore House Price Prediction (Multivariate Linear Regression + PCA)

### 📊 Objective
Build a **multivariate linear regression** model to predict house prices in Bangalore using real-world features like size, location, area type, etc. Apply PCA to reduce dimensionality and visualize the impact.

---

### 🧩 Dataset Used
- **Source**: Bengaluru_House_Data.csv
- **Features**: area_type, availability, location, size, society, total_sqft, bath, balcony, price

---

### ⚙️ Project Steps

1. **Data Cleaning**
   - Removed rows with missing critical fields
   - Extracted `bhk` from size
   - Cleaned `total_sqft` (handled ranges like '2100 - 2850')
   - Removed outliers (e.g., sqft per bhk < 300, excessive bathrooms)

2. **Feature Engineering**
   - Encoded categorical fields: `location`, `area_type`
   - Added binary `availability_flag` for "Ready to move"
   - Scaled all numeric features using `StandardScaler`

3. **Model Building**
   - Trained a **LinearRegression** model using all 1222 features
   - Evaluated using **RMSE** and **R²**

4. **PCA Analysis**
   - Applied PCA with 2 and 3 components
   - Visualized using 2D and 3D scatter plots
   - Printed **explained variance** and **top contributing features**

5. **Regression with PCA**
   - Trained and evaluated models using PCA-reduced features (2 and 3 components)
   - Visualized actual vs predicted + residuals

---

### 📈 Final Results

| Model Type              | RMSE     | R² Score | Variance Explained |
|-------------------------|----------|----------|---------------------|
| Full Feature Regression | 123.76   | 0.105    | —                   |
| PCA Regression (2 PC)   | 104.02   | 0.368    | 0.41%               |
| PCA Regression (3 PC)   | 103.45   | 0.375    | 0.53%               |

---

### 🔍 Top PCA Features

| PC | Top Features                                             |
|----|-----------------------------------------------------------|
| 1  | bhk, bath, total_sqft, Plot Area, Super built-up Area     |
| 2  | Super built-up Area, Plot Area, balcony, bath, total_sqft |
| 3  | availability_flag, balcony, Carpet Area, Hosa Road, Whitefield |

---

### 📊 Visualizations
- ✅ Actual vs Predicted Prices
- ✅ Residual Distributions
- ✅ PCA 2D and 3D Plots
- ✅ Feature contributions to each PCA axis

---

### 🧠 Conclusion
- PCA significantly improved model performance despite explaining only ~0.5% of the original variance.
- The top features influencing price were `bhk`, `bath`, `sqft`, and key locations.
- PCA reduced noise and helped the model generalize better, reducing RMSE by over 16% compared to the full model.

---

### ✅ Tools Used
- Python 3.9.13
- scikit-learn
- pandas, matplotlib, seaborn
- PCA, LinearRegression, StandardScaler