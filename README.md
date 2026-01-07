# Property-Price-Valuation

## 📊 Exploratory Data Analysis (EDA)

This section documents the exploratory analysis conducted to understand the structure, quality, and statistical properties of the dataset before model building. The objective of the EDA is to identify data issues, assess distributional behavior, examine relationships across variables, and guide feature engineering and model selection.

---

### 1. Data Overview

- The dataset consists of **structured tabular observations** with a mix of:
  - Continuous numerical variables (e.g., price-related and transformed features)
  - Discrete/categorical attributes (e.g., location-based identifiers)
  - Temporal fields (date/time variables)

- Initial inspection confirms:
  - No duplicate rows
  - Consistent dimensionality across observations
  - Reasonable ranges for most numerical features

---

### 2. Data Quality and Cleaning

#### 2.1 Missing Values
- Most variables are **well-behaved**, with:
  - No major missing-value clusters
  - No variables dominated by zeros or placeholder values
- Missingness, where present, appears **random and sparse**, not systematic.

#### 2.2 Date Correction
- The date variable required **format correction and standardization**
- Once corrected, temporal ordering and indexing behaved as expected

---

### 3. Categorical Variables

- A limited number of categorical features are present
- These variables:
  - Have low to moderate cardinality
  - Do not exhibit extreme imbalance
- **One-Hot Encoding** is identified as the appropriate transformation strategy for downstream models

---
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/2839c47c-80c8-43b4-8a8d-4522e56b4a0d" />
<img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/db4ca14c-ea8f-437a-af3b-a58a117d66dd" />


### 4. Univariate Analysis

#### 4.1 Distributional Properties
- Most numerical features display:
  - Mild skewness
  - Reasonable variance
- A few variables exhibit **strong right skew**, motivating:
  - Logarithmic
  - Square-root
  - Box–Cox–style transformations

#### 4.2 Target Variable Behavior
- The target variable (price):
  - Is continuous and strictly positive
  - Exhibits no abnormal spikes or gaps
  - Has no excessive zero inflation
 
<img width="1490" height="1189" alt="image" src="https://github.com/user-attachments/assets/610bdbbe-4747-4bc9-aebd-7c8aa9a58923" />


---

### 5. Multivariate Analysis

#### 5.1 Correlation Structure
- Pairwise correlation analysis reveals:
  - No severe multicollinearity across raw features
  - Groups of highly correlated transformed variables (e.g., square-root and log variants)
- These correlations are **expected and intentional**, stemming from feature engineering rather than data flaws

<img width="695" height="616" alt="image" src="https://github.com/user-attachments/assets/93e46ac6-79f2-4d7b-9c03-f8c4b1bdeff3" />


#### 5.2 Feature Redundancy
- While some engineered features are strongly correlated:
  - This redundancy can be addressed during:
    - Regularization
    - Feature selection
    - Dimensionality reduction (if required)

---
<img width="2701" height="2701" alt="image" src="https://github.com/user-attachments/assets/6cf1acf3-2740-4cc5-a84b-752730614067" />
<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/bc40213f-9b32-459b-afe8-13015aff6c76" />


### 6. Spatial and Location-Based Analysis

- Zip code / location identifiers show:
  - No strong standalone linear relationship with price
  - Weak but structured interaction effects
- This suggests:
  - Location matters indirectly
  - Nonlinear or cluster-based spatial methods may be more appropriate

---
<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/1c656106-112c-4265-815a-fc46f2d110ed" />


### 7. Temporal Analysis

- Time-series diagnostics indicate:
  - No strong autocorrelation in price
  - No persistent trends or seasonal shocks
- The data behaves as:
  - Cross-sectional with weak temporal dependence
- This supports treating the dataset as **pooled observations** rather than a strict time-series problem

---
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/589192a2-47bd-4b44-8085-91bfd2a6c75b" />


### 8. Clustering and Structure Discovery

- Preliminary clustering diagnostics suggest:
  - Non-spherical, density-based groupings
  - Presence of local clusters rather than global partitions
- **DBSCAN** is identified as a suitable clustering method due to:
  - Its robustness to noise
  - No requirement to pre-specify the number of clusters

---
<img width="1222" height="528" alt="image" src="https://github.com/user-attachments/assets/a4404531-6699-4d43-b09d-2ce9e79c7863" />

<img width="2891" height="551" alt="image" src="https://github.com/user-attachments/assets/5e60cbc5-23f5-4ba0-867a-b1c06e51e743" />

### 9. Key EDA Insights

- The dataset is **clean, well-curated, and model-ready**
- No major violations of classical modeling assumptions
- Feature transformations meaningfully improve symmetry and scale
- Limited multicollinearity and stable variance structure
- Spatial and temporal dimensions are subtle but informative

---

### 10. Implications for Modeling

Based on EDA findings:
- Tree-based models and regularized regressions are suitable
- Transformed features should be retained selectively
- Spatial clustering can enhance predictive performance
- No aggressive imputation or anomaly correction is required

---
<img width="937" height="547" alt="image" src="https://github.com/user-attachments/assets/b1bfa28d-cff3-4903-b659-58bd5939aa5b" />
<img width="1621" height="701" alt="image" src="https://github.com/user-attachments/assets/2a2518e5-8ed7-42c3-aab7-347e86067eab" />


### 11. Reproducibility

All EDA steps are:
- Fully reproducible
- Implemented in the accompanying notebook
- Executable without external preprocessing

Refer to `data_EDA.ipynb` for full code, plots, and diagnostics.

---
