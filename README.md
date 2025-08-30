# 💳 Credit Card Fraud Detection Using Machine Learning

# 1. Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. Fraudulent transactions are rare but financially damaging, making early detection crucial for banking security.
I use the Kaggle Credit Card Fraud dataset, which contains anonymized features (V1–V28, Time, Amount) and a target label Class (0 = legitimate, 1 = fraud).

The system provides two functionalities:

Batch Prediction → Analyze multiple transactions at once
Single Transaction Prediction → Interactively analyze individual transactions

# 2. Installation & Run
# Clone repository
git clone https://github.com/<username>/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the Streamlit app
streamlit run app.py

The application will be available in your browser at:
  Local URL: http://localhost:8501
  Network URL: http://192.168.0.235:8501


# Dataset
Download the dataset from **[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
Place the Kaggle dataset (creditcard.csv) inside the data/ folder before running the project.

# 3. Dataset
Source: Kaggle – Credit Card Fraud Detection
Features:
V1–V28: anonymized PCA components
Time, Amount: transaction metadata
Target: Class (0 = legitimate, 1 = fraud)
Class Imbalance: ~99.8% legitimate vs. ~0.2% fraud

# 4. Data Preprocessing
Scaling: Time and Amount scaled using StandardScaler
Feature Order: Inputs arranged as [Time, V1–V28, Amount]
Train-Test Split: 80/20 split for training and evaluation
Handling Imbalance: Used ROC AUC and PR AUC instead of accuracy

# 5. Model Training
Algorithm: RandomForestClassifier
Why RandomForest? Robust to imbalance, handles non-linear patterns, outputs probability scores
Steps:
Scale Time and Amount
Train/test split
Train RandomForest
Evaluate with metrics
Save artifacts:
model.joblib → trained model
preprocess.joblib → scaler
metadata.json → threshold
Example Metrics:
Accuracy: 1.00
ROC AUC: 0.976
PR AUC: 0.873
F1-score (fraud class): 0.83

# 6. Streamlit Dashboard
The interactive Streamlit app (app.py) provides:
Batch Prediction: Upload a CSV → model predicts and classifies transactions
Single Prediction: Pre-filled random transaction → fraud probability + prediction (LEGIT or FRAUD)

# 7.Folder Structure & Screenshots
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv           # Kaggle dataset
│
├── models/
│   ├── model.joblib             # Trained RandomForest model
│   ├── preprocess.joblib        # Scaler for Time & Amount
│   └── metadata.json            # Threshold and other info
│
├── train_model.py               # Train model & save artifacts
├── app.py                       # Streamlit app for demo
└── README.md                    # Project documentation

**[Dashboard Screenshot](https://github.com/sumit48/Credit-Card-Fraud-Detection/Project Demo/dashboard.png)**

# 8. Conclusion

This project shows how machine learning can strengthen financial cybersecurity by effectively detecting fraudulent credit card transactions, even in highly imbalanced datasets. Using a RandomForest model with proper preprocessing and evaluation (ROC AUC, PR AUC), the system achieves strong fraud detection performance.
From a security standpoint, it demonstrates how data-driven models can complement traditional rule-based systems, enhancing fraud prevention strategies. The interactive Streamlit app makes the solution practical for both technical and non-technical users, bridging the gap between machine learning research and real-world cybersecurity defense in financial systems.
