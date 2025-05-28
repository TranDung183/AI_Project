import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dữ liệu
data = pd.read_csv("C:/CODE/Python/AI_doAn/heart_cleveland_upload.csv")
target_column = data.columns[-1]
X = data.drop(columns=[target_column])
y = data[target_column]

# 2. Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Trọng số xác suất động (giả lập hoặc bạn có thể thay bằng logic y sinh)
np.random.seed(42)
sample_weights = np.random.rand(len(y_train))

# 4. Huấn luyện mô hình Elastic Net
model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
    alphas=[0.01, 0.1, 1, 10],
    cv=5,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# 5. SMART Attribute Ranking
coef_abs = np.abs(model.coef_)
coef_sum = coef_abs.sum()
if coef_sum != 0:
    normalized_weights = coef_abs / coef_sum
else:
    normalized_weights = coef_abs  # tránh chia cho 0

attribute_ranking = pd.Series(normalized_weights, index=X.columns).sort_values(ascending=False)
print("\n=== SMART Attribute Ranking ===")
print(attribute_ranking)

# 6. Tính xác suất theo thời gian (timeline)
def compute_timeline_prob(prob_array, decay=0.85, steps=10):
    return np.clip(np.sum([prob_array * (decay**t) for t in range(steps)], axis=0), 0, 1)

y_pred_prob = model.predict(X_test)
timeline_probs = compute_timeline_prob(y_pred_prob)

# 7. Phân loại theo TUP
def classify_by_tup(tup):
    if tup <= 0.01:
        return 0  # Không bệnh
    elif 0.01 < tup <= 0.05:
        return 1  # Nguy cơ thấp
    elif 0.05 < tup <= 0.09:
        return 2  # Nguy cơ trung bình
    else:
        return 3  # Nguy cơ cao

y_pred_class = [classify_by_tup(p) for p in timeline_probs]
y_true_class = [classify_by_tup(p) for p in compute_timeline_prob(y_test)]

# 8. Đánh giá và biểu đồ
print("\n=== Confusion Matrix ===")
conf_matrix = confusion_matrix(y_true_class, y_pred_class)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n=== Classification Report ===")
print(classification_report(y_true_class, y_pred_class))

# 9. Feature Weights trực quan
feature_weights = pd.Series(model.coef_, index=X.columns)
print("\n=== Elastic Net Coefficients ===")
print(feature_weights.sort_values(ascending=False))

feature_weights.sort_values().plot(kind='barh', title="Feature Importance (EPERM)")
plt.tight_layout()
plt.show()
