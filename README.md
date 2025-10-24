# 🏆 American Express Decision Science Campus Challenge 2025  
### 3nd Runner-Up Winning Solution

This repository contains the **3rd Runner-Up** solution for the **American Express Decision Science Campus Challenge 2025**.  
The project focused on **predicting customer click behavior on offers**, framed as a **ranking problem**.

Our final solution achieved a **MAP@7 score of 80.09%** on a **temporally-split validation set**.

---

## 👨‍💻 Team Members
- **Raju Gupta**  
- **Priyanshu Kumar**  
- **Mridul Singh**  
*(Indian Institute of Technology (ISM) Dhanbad)*

---

## 🚀 Final Solution Overview

Our winning pipeline was built on three core pillars:

1. **Strategic Feature Engineering** – Powerful time-based and offer-interaction features.  
2. **Robust Validation Strategy** – Strict temporal (chronological) split to mimic real-world deployment and avoid data leakage.  
3. **Rank-Optimized Modeling** – Leveraging `XGBRanker` to directly optimize ranking metrics such as MAP@7.

---

## 🧠 Methodology

### 1. Feature Engineering & Selection

We engineered a robust set of **339 features** to capture user behavior and offer characteristics.

#### 🔹 Key Engineered Features

**Time-Based Features**
- `time_until_next_event`: Time remaining until the customer's next interaction  
- `time_since_last_event`: Time elapsed since the customer's last interaction  
- `time_since_session_start`: Duration since the start of the current session  

**Offer Interaction Metrics**
- `offer_ctr`: Click-through rate of the specific offer  
- `offer_total_interactions`: Total number of interactions with the offer  
- `offer_total_positives`: Total clicks received for the offer  

#### 🔹 Feature Selection Process
- **Dropped High-Null Columns:** Removed features with >90% missing values  
- **Removed Leaky Features:** Excluded features with abnormally high ROC-AUC scores against the target  
- **Post-Pruning:** Retained only features that improved MAP@7

---

### 2. Sampling & Validation Strategy

This was the **most critical component** of our solution.

#### 🕒 Technique: Temporal Sampling
**Implementation:**  
The dataset was sorted chronologically based on event timestamp (`id4`).

| Split | Percentage | Rows |
|:------|:------------|:------|
| **Training Set** | 90% | 693,147 |
| **Validation Set** | 10% | 77,017 |

**Reasoning:**  
Ensures the model learns from the past to predict the “future,” eliminating data leakage and providing a realistic performance estimate.

---

### 3. Modeling

We framed the task as a **Learning-to-Rank (LTR)** problem.

**Final Model:** `XGBRanker`

#### Why XGBRanker?
While we experimented with `XGBoost` and `LightGBM` classifiers, `XGBRanker` consistently outperformed them by directly optimizing for **pairwise ranking loss**, aligning perfectly with **MAP@7**.

---

## 📊 Performance & Results

| Model Iteration | Description | MAP@7 (Validation) |
|:----------------|:-------------|:------------------:|
| 1 | XGBoost (Baseline) | 48.60% |
| 2 | XGBRanker + Feature Engineering | 62.14% |
| 3 | XGBRanker + FE + Temporal Sampling | **80.09%** |

---

## 🔍 Top 10 Most Important Features

| Rank | Feature | Importance (by Gain) % |
|:----:|:---------|:---------------------:|
| 1 | `time_until_next_event` | 5.83 |
| 2 | `time_since_last_event` | 3.63 |
| 3 | `f210` | 2.37 |
| 4 | `offer_ctr` | 2.10 |
| 5 | `f206` | 1.05 |
| 6 | `f314` | 0.98 |
| 7 | `f207` | 0.97 |
| 8 | `f209` | 0.95 |
| 9 | `f358` | 0.91 |
| 10 | `f127` | 0.76 |

---

## 🔮 Future Improvements

We identified several directions for enhancement:

- **Model Ensembling:** Combine `XGBRanker`, `LightGBM`, and `CatBoost` for better robustness  
- **Integration of Additional Datasets:** Use the provided external datasets to enrich the feature space  
- **Data Fusion Strategies:** Develop temporal-consistent fusion techniques for multi-source data  

---
