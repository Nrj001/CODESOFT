
---

## 📁 2. `Credit_Card_Fraud_Detection/README.md`

```markdown
# 💳 Credit Card Fraud Detection Using Machine Learning

This project detects fraudulent credit card transactions using SMOTE for balancing and Random Forest for classification.

## 📌 Overview
In real-world banking systems, class imbalance is a major issue. Fraudulent transactions are rare, so we use SMOTE to balance the dataset and build a reliable model.

## 🔧 Tools & Libraries
- Python
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)

## 📊 Features (if using real dataset)
- Time
- Amount
- V1–V28 features (from PCA transformation)
- Target: Class (0 = genuine, 1 = fraud)

## 🧠 Model Used
- Random Forest Classifier
- SMOTE for data balancing

## 📋 Metrics
- Confusion Matrix
- Precision, Recall, F1-Score
- Accuracy

## 🛠 How to Run
1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
