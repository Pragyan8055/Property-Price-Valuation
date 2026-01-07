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
## ⚙️ Data Preprocessing

This section documents the preprocessing pipeline applied to transform the raw dataset into a statistically coherent, model-ready format. Each step is motivated by econometric validity, numerical stability, and downstream modeling requirements.

---

### 1. Objectives of Preprocessing

The preprocessing stage is designed to:

- Enforce consistency across variable scales and distributions  
- Reduce skewness and heteroskedasticity  
- Eliminate redundant or weakly informative features  
- Encode categorical information without inducing spurious ordering  
- Preserve interpretability while enabling flexible model classes  

---

### 2. Feature Selection and Pruning

- Raw features with:
  - Near-zero variance
  - Redundant semantic meaning
  - Perfect or near-perfect correlation  
  were removed or consolidated.

- Transformed variables were retained **only when they added distributional or numerical benefits**, not for mechanical expansion of the feature space.

**Insight:**  
This balances *bias–variance trade-off* by preventing overparameterization while preserving nonlinear structure.

---

### 3. Handling Skewness and Scale

#### 3.1 Nonlinear Transformations
- Highly skewed numerical variables were transformed using:
  - Logarithmic
  - Square-root
- Transformations were chosen based on:
  - Skewness reduction
  - Preservation of monotonicity

**Interpretation:**  
These transformations stabilize variance and improve convergence behavior in optimization-based models.

#### 3.2 Scaling and Normalization
- Continuous features were standardized to zero mean and unit variance
- Scaling was applied **after transformation**, ensuring comparability across features

**Why this matters:**  
Standardization is critical for:
- Regularized regressions (LASSO, Ridge)
- Distance-based methods
- Gradient-based optimization

---

### 4. Categorical Encoding

- Nominal categorical variables were encoded using **One-Hot Encoding**
- No ordinal assumptions were imposed
- High-cardinality categories were monitored for sparsity effects

**Design choice:**  
One-Hot Encoding preserves interpretability and avoids artificial ranking distortions.

---

### 5. Temporal Variable Processing

- Date variables were:
  - Parsed and standardized
  - Converted to numeric representations where needed
- Temporal fields were not treated as time-series drivers due to:
  - Weak autocorrelation
  - Absence of strong trend or seasonality

**Implication:**  
The dataset is treated as **pooled cross-sectional**, not dynamic time-series data.

---

### 6. Outlier and Influence Control

- No aggressive trimming or winsorization was applied
- Extreme observations were retained unless:
  - They violated logical constraints
  - They resulted from data entry errors

**Rationale:**  
Preserving tail observations is essential for realistic price and heterogeneity modeling.

---

### 7. Multicollinearity Management

- Correlated transformed features were:
  - Explicitly identified
  - Left intact where models can handle collinearity internally

**Strategy:**  
Collinearity is addressed at the *modeling stage* (regularization, tree splits), not via premature feature deletion.

---

### 8. Final Feature Matrix Construction

The final dataset consists of:

- Transformed and standardized numerical variables
- One-hot encoded categorical indicators
- Cleaned and aligned temporal fields

All features are:
- Numeric
- Scale-consistent
- Free of missing values

---

### 9. Output Artifacts

- Preprocessed feature matrix ready for modeling
- Transformation logic fully reproducible
- No information leakage across stages

Refer to `preprocessing.ipynb` for the full implementation.

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/1443dce6-5ee0-4308-bc72-ab608e280030" />

---

### 10. Modeling Readiness Summary

| Criterion                     | Status |
|------------------------------|--------|
| Missing values               | None   |
| Scale consistency            | ✔️     |
| Skewness controlled          | ✔️     |
| Multicollinearity monitored  | ✔️     |
| Model compatibility          | ✔️     |

---

## 🤖 Modeling Framework

This section describes the dual-model strategy adopted for price prediction, detailing the mathematical structure of each model, the training procedure, and the logic behind combining their outputs for final price estimation.

---

### 1. Modeling Strategy Overview

Two complementary models are trained:

1. **Model A: Pretrained Neural Network Regressor**
2. **Model B: Tree-Based Gradient Boosting Regressor**

The motivation for this architecture is to combine:
- **Global nonlinear function approximation** (Model A)
- **Local interaction discovery and robustness to heterogeneity** (Model B)

This hybrid strategy improves predictive accuracy, stability, and computational efficiency.

---

### 2. Model A: Pretrained Neural Network

#### 2.1 Model Structure

Model A is a feedforward neural network initialized using **pretrained weights** obtained from prior training on a large, related dataset.

Let the feature vector be \( x \in \mathbb{R}^p \).

The network implements:

\[
\hat{y}_A = f_\theta(x) = W_L \sigma \left( \cdots \sigma(W_1 x + b_1) \cdots \right) + b_L
\]

where:
- \( \sigma(\cdot) \) is a nonlinear activation function
- \( \theta = \{W_l, b_l\}_{l=1}^L \) are learnable parameters

---

#### 2.2 Loss Function

The model is trained using **Mean Squared Error (MSE)**:

\[
\mathcal{L}_A(\theta) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_{A,i})^2
\]

---

#### 2.3 Role of Pretraining

Using pretrained weights provides:

- Faster convergence:
\[
\|\nabla \mathcal{L}_A^{(0)}\| \ll \|\nabla \mathcal{L}_A^{\text{random}}\|
\]

- Improved generalization via learned feature hierarchies
- Reduced risk of overfitting on limited data

**Key Insight:**  
Pretraining shifts optimization from *representation learning* to *task adaptation*, reducing both training time and variance.

---

### 3. Model B: Gradient Boosting Regressor

#### 3.1 Model Structure

Model B is an additive ensemble of regression trees:

\[
\hat{y}_B = \sum_{m=1}^{M} \gamma_m h_m(x)
\]

where:
- \( h_m(x) \) is a decision tree
- \( \gamma_m \) is the learning rate–scaled contribution

---

#### 3.2 Loss Function

The boosting procedure minimizes:

\[
\mathcal{L}_B = \sum_{i=1}^N (y_i - \hat{y}_{B,i})^2
\]

via functional gradient descent:

\[
h_m = \arg\min_h \sum_{i=1}^N \left( r_{i}^{(m)} - h(x_i) \right)^2
\]

with residuals:

\[
r_{i}^{(m)} = y_i - \hat{y}_{B,i}^{(m-1)}
\]

---

#### 3.3 Salient Properties

- Automatically captures nonlinear interactions
- Robust to multicollinearity
- Requires minimal feature scaling
- Performs well under heterogeneous regimes

**Interpretation:**  
This model excels at *local structure discovery* and correcting systematic errors left by smooth approximators.

---

### 4. Model Complementarity

| Dimension              | Model A (Neural Net) | Model B (Boosting) |
|------------------------|----------------------|--------------------|
| Global smoothness      | ✔️                   | ❌                 |
| Local interactions     | ❌                   | ✔️                 |
| Data efficiency        | Medium               | High               |
| Interpretability       | Low                  | Medium             |
| Training speed         | Fast (pretrained)    | Moderate           |

---

### 5. Model Combination Strategy

Final price predictions are obtained via **weighted ensembling**:

\[
\hat{y}_{\text{final}} = \alpha \hat{y}_A + (1 - \alpha) \hat{y}_B
\]

where:
\[
\alpha \in [0,1]
\]

is chosen via validation performance.

---

### 6. Theoretical Motivation for Ensembling

The ensemble reduces expected generalization error:

\[
\mathbb{E}[(y - \hat{y}_{\text{final}})^2] 
= \alpha^2 \sigma_A^2 + (1-\alpha)^2 \sigma_B^2 + 2\alpha(1-\alpha)\text{Cov}(A,B)
\]

When prediction errors are weakly correlated:
\[
\text{Cov}(A,B) \approx 0
\]

the ensemble strictly dominates individual models.

---

### 7. Training Procedure

1. Train Model A using pretrained initialization
2. Fine-tune on project-specific data
3. Train Model B independently on the same feature matrix
4. Optimize ensemble weight \( \alpha \) on validation set
5. Generate final predicted prices

---

### 8. Output and Interpretation

- Individual model predictions are retained for diagnostics
- Final prices reflect:
  - Smooth global trends (Model A)
  - Local corrections (Model B)

This yields stable, accurate, and economically plausible price estimates.

---

### 9. Reproducibility

All modeling steps are:
- Deterministic given random seeds
- Fully documented in `model_training.ipynb`
- Independent of data leakage

---
