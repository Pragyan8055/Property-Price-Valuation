# Property-Price-Valuation

# 📈 Neural Network–Based Price Prediction

## Introduction

Accurate price estimation in high-dimensional environments is a central problem in applied economics, data science, and computational decision-making. Traditional parametric approaches often struggle to accommodate nonlinearities, interaction effects, and heterogeneous responses across observations, particularly when the feature space is large and structurally complex.

This project develops a **purely neural network–based framework** for price prediction, emphasizing robustness, interpretability of modeling choices, and computational efficiency. The approach combines rigorous data preparation with modern deep learning techniques, leveraging **pretraining and model ensembling** to improve generalization while reducing training instability.

---

### Motivation

Price formation processes are rarely linear. They are influenced by:
- Nonlinear interactions between observable characteristics
- Latent structural factors that are difficult to specify ex ante
- Heterogeneity across locations and contexts

Neural networks provide a flexible function approximation framework capable of capturing such complexity. However, naïvely training deep models can lead to:
- Overfitting
- Slow convergence
- Sensitivity to initialization and scaling

This project addresses these challenges by integrating **exploratory data analysis, disciplined preprocessing, representation learning through pretraining, and controlled fine-tuning** within a unified pipeline.

---

### Methodological Contributions

The key methodological features of the project are:

- A structured **EDA pipeline** to diagnose distributional, spatial, and temporal properties of the data  
- A **statistically motivated preprocessing stage** that controls skewness, scale, and redundancy  
- A **two-stage neural network architecture**, separating representation learning from price calibration  
- The use of **pretrained neural networks** to improve optimization stability and reduce training time  
- A **model combination strategy** that balances global nonlinear structure with local predictive accuracy  

Unlike many applied machine learning pipelines, this project **exclusively relies on neural networks**, avoiding tree-based or boosting methods in order to maintain architectural consistency and focus on representation learning.


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

## 🧠 Modeling Framework

This section describes the neural network–based modeling strategy used to estimate final prices. Two complementary neural architectures are trained and combined to exploit both representation learning and task-specific calibration, while maintaining computational efficiency and numerical stability.

---

### 1. Modeling Objectives

The modeling stage is designed to:

- Capture nonlinear relationships between features and prices
- Exploit transfer learning to improve generalization
- Reduce training time and overfitting risk
- Combine structural and predictive strengths across models

Only **neural network models** are used in this project.

---

### 2. Overview of the Two-Model Strategy

Two neural networks are trained:

1. **Pretrained Neural Network (Base Model)**  
   - Learns general feature representations
   - Acts as a high-capacity nonlinear function approximator
   - Provides stable, transferable embeddings

2. **Task-Specific Neural Network (Fine-Tuned Model)**  
   - Trained on top of pretrained representations
   - Focuses on price calibration and local heterogeneity
   - Improves accuracy and robustness

The final price prediction is obtained by **combining the outputs** of both models.

---


### 3. Model 1: Task-Specific Neural Network

#### 3.1 Architecture

The second neural network maps learned representations to prices:

\[
\hat{y}_i^{(2)} = g_{\phi}(\mathbf{h}_i)
\]

where:
- \( g_{\phi}(\cdot) \) is a shallow neural network
- \( \phi \) are trainable parameters
- \( \hat{y}_i^{(2)} \) is the predicted price

This model:
- Has fewer layers
- Emphasizes calibration rather than representation learning
- Is trained with the pretrained layers frozen or partially unfrozen

---

#### 3.2 Why a Second Model?

- Separates **representation learning** from **price estimation**
- Improves interpretability of training dynamics
- Enables controlled fine-tuning
- Reduces variance in final predictions

---
<img width="1024" height="1024" alt="Gemini_Generated_Image_smddupsmddupsmdd" src="https://github.com/user-attachments/assets/aff05670-5ff6-42a9-9c35-2576c7cc6f39" />


### 4. Model 2: Pretrained Neural Network

#### 4.1 Architecture

Let \( \mathbf{x}_i \in \mathbb{R}^d \) denote the preprocessed feature vector for observation \( i \).

The pretrained neural network learns a mapping:

\[
\mathbf{h}_i = f_{\theta}(\mathbf{x}_i)
\]

where:
- \( f_{\theta}(\cdot) \) is a deep feedforward neural network
- \( \mathbf{h}_i \in \mathbb{R}^k \) is a latent representation
- \( \theta \) denotes pretrained parameters

The network consists of:
- Fully connected layers
- Nonlinear activations (ReLU)
- Dropout regularization

---

#### 4.2 Motivation for Pretraining

Using a pretrained model:

- Improves **initialization quality**
- Reduces convergence time
- Stabilizes optimization in high-dimensional spaces
- Mitigates overfitting under limited data

**Key insight:**  
Pretraining allows the model to learn **generic nonlinear structure** before task-specific fine-tuning.

---
<img width="1408" height="768" alt="Gemini_Generated_Image_fe08life08life08" src="https://github.com/user-attachments/assets/335700fe-0b0b-4d0c-80a4-6729ba7111e6" />


### 5. Loss Functions

Both models are trained using **Mean Squared Error (MSE)** loss.

#### 5.1 Mean Squared Error

For \( N \) observations:

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

where:
- \( y_i \) is the observed price
- \( \hat{y}_i \) is the predicted price

MSE is chosen because:
- It penalizes large errors strongly
- It aligns with continuous price estimation
- It has well-behaved gradients

---

### 6. Optimization

- Parameters are optimized using **Adam**
- Learning rate scheduling is applied
- Early stopping prevents overfitting

Formally, parameters are updated as:

\[
\theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} \mathcal{L}
\]

where \( \eta \) is the adaptive learning rate.

---

### 7. Model Combination Strategy

Final prices are computed by **combining predictions** from both neural networks.

Let:
- \( \hat{y}_i^{(1)} \) be the prediction from the pretrained model
- \( \hat{y}_i^{(2)} \) be the prediction from the fine-tuned model

The final predicted price is:

\[
\hat{y}_i = \alpha \hat{y}_i^{(1)} + (1 - \alpha) \hat{y}_i^{(2)}, \quad \alpha \in [0,1]
\]

where \( \alpha \) is chosen via validation performance.

---

### 8. Why Model Combination Improves Performance

- Reduces prediction variance
- Balances global structure and local fit
- Acts as an implicit regularization mechanism
- Improves out-of-sample robustness

**Interpretation:**  
The pretrained model captures broad nonlinear structure, while the fine-tuned model corrects systematic biases.

---

### 9. Final Outputs

The modeling stage produces:

- Individual model predictions
- Combined final price estimates
- Training and validation loss diagnostics

All modeling steps are fully reproducible and implemented in `model_training.ipynb`.

---

### 10. Modeling Summary

| Aspect                     | Description |
|----------------------------|-------------|
| Model class                | Neural Networks only |
| Pretraining                | Yes |
| Loss function              | MSE |
| Optimization               | Adam, Huber |
| Model combination          | Weighted ensemble |
| Output                     | Final predicted prices |

---
