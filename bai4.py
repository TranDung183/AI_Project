# PMIA Prediction with 12 Classifiers and SHAP Explanation

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import BorderlineSMOTE
from boruta import BorutaPy
import shap
import warnings
warnings.filterwarnings('ignore')

# Load your dataset (replace with actual path)
df = pd.read_csv("myocardial_infarction_data.csv")

df = df.dropna(axis=1, thresh=int(0.1 * len(df)))  # Remove >90% missing
X = df.drop('PMIA', axis=1)
y = df['PMIA']

# Impute missing values
for col in X.columns:
    null_ratio = X[col].isnull().mean()
    imputer = SimpleImputer(strategy='mean' if null_ratio < 0.3 else 'most_frequent')
    X[col] = imputer.fit_transform(X[[col]])

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance data
smote = BorderlineSMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Feature Selection using Boruta
rf_for_boruta = RandomForestClassifier(n_estimators=100, random_state=42)
boruta = BorutaPy(rf_for_boruta, n_estimators='auto', verbose=0, random_state=42)
boruta.fit(X_res, y_res)
X_boruta = boruta.transform(X_res)
selected_features = np.array(X.columns)[boruta.support_]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_boruta, y_res, test_size=0.2, random_state=42)

# Define model functions
def get_svm():
    return SVC(probability=True, random_state=42)

def get_knn():
    return KNeighborsClassifier()

def get_tree():
    return DecisionTreeClassifier(random_state=42)

def get_qda():
    return QuadraticDiscriminantAnalysis()

def get_xgb():
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

def get_lgb():
    return lgb.LGBMClassifier(random_state=42)

def get_voting():
    return VotingClassifier(estimators=[
        ('svm', SVC(probability=True)), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier())], voting='soft')

def get_bagging():
    return BaggingClassifier(random_state=42)

def get_adaboost():
    return AdaBoostClassifier(random_state=42)

def get_gbdt():
    return GradientBoostingClassifier(random_state=42)

def get_rf():
    return RandomForestClassifier(random_state=42)

def get_stacking():
    return StackingClassifier(estimators=[
        ('rf', RandomForestClassifier()), ('svm', SVC(probability=True))], final_estimator=LogisticRegression())

# Classifier mapping
classifiers = {
    'SVM': get_svm(),
    'KNN': get_knn(),
    'Tree': get_tree(),
    'QDA': get_qda(),
    'XGB': get_xgb(),
    'LGB': get_lgb(),
    'Voting': get_voting(),
    'Bagging': get_bagging(),
    'AdaBoost': get_adaboost(),
    'GBDT': get_gbdt(),
    'RF': get_rf(),
    'Stacking': get_stacking()
}

# Evaluation metrics
results = []
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    scores = cross_val_score(model, X_train, y_train, cv=5)
    results.append({
        'Model': name,
        'CV-ACC': f"{scores.mean():.2%} ± {scores.std():.2%}",
        'ACC': accuracy_score(y_test, y_pred),
        'PREC': precision_score(y_test, y_pred),
        'REC': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print(results_df)

# SHAP Explanation for Best Model (SVM assumed best)
explainer = shap.Explainer(classifiers['SVM'], X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=selected_features)
