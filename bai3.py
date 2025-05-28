import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ---- 1. Đọc dữ liệu ----
# Thay 'your_data.csv' bằng tên file dữ liệu của bạn
df = pd.read_csv('your_data.csv')  # file dữ liệu từ Alibaba Tianchi (đã được gán nhãn 'churn')

# ---- 2. Tiền xử lý ----
# Giả sử cột nhãn là 'churn', thay thế giá trị thiếu = 0
df.fillna(0, inplace=True)
X = df.drop(columns=['churn'])
y = df['churn']

# ---- 3. Chia dữ liệu train/test ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 4. Hàm đánh giá mô hình ----
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n--- {name} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    if y_proba is not None:
        print("AUC:", roc_auc_score(y_test, y_proba))

# ---- 5. Huấn luyện & đánh giá các mô hình ----
# (1) Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model(dt_model, "Decision Tree")

# (2) Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf_model, "Random Forest")

# (3) Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
evaluate_model(gb_model, "Gradient Boosting")
