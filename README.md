# Bank Term Deposit Prediction — Machine Learning Final Project

## 1. Project idea
This project predicts whether a bank client will subscribe to a term deposit after a marketing phone campaign.

The task is a **binary classification** problem:

- `yes` — the client subscribed to a term deposit
- `no` — the client did not subscribe

This project uses the **Bank Marketing Dataset** from the **UCI Machine Learning Repository**, not Kaggle and not a synthetic dataset.

## 2. Why this dataset is valid
The project guide says the dataset must be a real, non-synthetic dataset from an official downloadable source. This dataset is from the UCI Machine Learning Repository, an official public ML data repository.

Dataset source:
https://archive.ics.uci.edu/dataset/222/bank+marketing

UCI citation:
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## 3. Dataset description
The dataset contains information from direct marketing campaigns of a Portuguese banking institution. The campaign was based on phone calls.

Target column:

- `y`: whether the client subscribed to a term deposit

Examples of features:

- `age`
- `job`
- `marital`
- `education`
- `balance`
- `housing`
- `loan`
- `contact`
- `month`
- `campaign`
- `pdays`
- `previous`
- `poutcome`

## 4. Machine learning algorithms used
The project trains and compares two models:

1. **Logistic Regression**
   - Good baseline model
   - Easy to explain
   - Works well with encoded categorical features

2. **Random Forest Classifier**
   - More powerful model
   - Handles nonlinear relationships better
   - Gives feature importance

The final model is selected based on ROC-AUC score.

## 5. Evaluation metrics
Because this is a classification task, the project uses:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

ROC-AUC and F1-score are especially useful because the dataset is imbalanced: most clients do not subscribe.

## 6. Project structure

```text
bank_marketing_ml_project/
├── README.md
├── requirements.txt
├── run_project.py
├── src/
│   ├── download_data.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── Bank_Marketing_Project.ipynb
├── data/
│   └── raw/
├── outputs/
│   └── figures/
└── models/
```

## 7. How to run

### Step 1. Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2. Run the full project

```bash
python run_project.py
```

Or run the training script directly:

```bash
python src/train.py
```

The script will:

1. Download the dataset from UCI
2. Load and clean the data
3. Perform basic EDA
4. Train Logistic Regression and Random Forest
5. Evaluate both models
6. Save metrics, plots, and the best model

## 8. Output files
After running, check:

```text
outputs/metrics.json
outputs/classification_report.txt
outputs/figures/target_distribution.png
outputs/figures/confusion_matrix.png
outputs/figures/roc_curve.png
outputs/figures/feature_importance.png
models/best_model.joblib
```

## 9. Live demo explanation
For the live demo, run:

```bash
python run_project.py
```

Then explain:

1. The dataset is downloaded from UCI.
2. Categorical columns are transformed using OneHotEncoder.
3. Numeric columns are scaled using StandardScaler for Logistic Regression.
4. Two models are trained and compared.
5. The best model is selected by ROC-AUC.
6. The confusion matrix and classification report show how well the model predicts `yes` and `no`.

## 10. Limitations
- The dataset is based on one Portuguese bank, so results may not generalize to all countries or banks.
- Some features, such as call duration, may only be known after the call, so they may not be available before making a marketing decision.
- The target is imbalanced, because fewer clients subscribe than reject the offer.

## 11. Possible improvements
- Tune hyperparameters using GridSearchCV.
- Try XGBoost or LightGBM.
- Remove `duration` and test a more realistic pre-call prediction model.
- Use SHAP values for deeper model interpretation.
