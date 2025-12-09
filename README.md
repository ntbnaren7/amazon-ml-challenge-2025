# Amazon ML Challenge 2025: Smart Product Pricing

**Global Rank #816 | 80,000+ Registrations**  
🏆 **1st Place – Campus Level**

This repository contains the methodology and code used for our solution in the **Amazon ML Challenge 2025: Smart Product Pricing**, where we secured **Rank #816 on the public leaderboard** among **80,000+ registered participants**, and **1st place at the campus level**.

Our focus was on building a **strong, well-regularized machine learning pipeline** that balances **performance, interpretability, and robustness** under the **SMAPE evaluation metric**.

---

## 🏷 Team Details

- **Team Name:** *SENTINELS*  

---

## 🚀 Solution Overview

Our approach is a **feature-engineering + gradient boosting–based regression pipeline**, designed to handle the skewed price distribution and metric sensitivity of the challenge.

Rather than relying on heavy multimodal models, we focused on:

- Careful target transformations  
- Strong tabular modeling  
- Metric-aligned optimization  
- Aggressive validation and error analysis  

---

## 🧠 Key Challenges & Strategy

### 1. SMAPE Metric Sensitivity

SMAPE disproportionately penalizes errors on **low-priced items**. To address this:

- Optimized models using **MAE / L1 loss**, which provides stable gradients near small values  
- Monitored validation performance specifically on **low-price buckets**

---

### 2. Skewed Price Distribution

Product prices showed a strong **right skew**.

- Applied **logarithmic transformation (`np.log1p`)** on the target variable  
- Converted predictions back using **`np.expm1()`** during evaluation  

This stabilized training and aligned optimization with the SMAPE metric.

---

### 3. Feature Robustness

We emphasized:

- Feature normalization  
- Removing noisy and redundant attributes  
- Iterative feature selection driven by validation feedback  

---

## 🛠 Model Architecture

### Feature Processing

- Cleaned and structured catalog metadata  
- Handled missing values and inconsistencies  
- Applied normalization where required  

---

### Model

- **Primary Models:** Gradient Boosting Trees  
  - XGBoost  
  - LightGBM (experimented variants)  
- **Objective:** MAE on log-transformed prices  
- **Early stopping** used to prevent overfitting  

---

### Training Setup

- Cross-validation–driven experimentation  
- Validation split carefully chosen to reflect leaderboard behavior  
- Hyperparameters tuned **iteratively**, not brute-forced  

---

## 📊 Model Performance

Evaluation was performed on a validation split designed to mirror public leaderboard trends.

| Metric | Score |
|------|------|
| SMAPE | Consistent with Rank #816 public leaderboard |
| MAE (log-scale) | Stable across folds |
| Generalization | Strong on low-price samples |

---

## 🏁 Conclusion

Our solution demonstrates that **solid fundamentals still matter** in competitive machine learning:

- Metric-aligned loss selection  
- Proper target transformations  
- Strong boosting models  
- Careful validation  

Achieving a **top ~1% global rank** and **1st place at the campus level** reinforced the value of **disciplined modeling over unnecessary complexity**.
