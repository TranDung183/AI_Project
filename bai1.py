import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Hàm kiểm tra và đọc dữ liệu
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_cols = ['title', 'bedrooms', 'land_size', 'living_size', 'location', 'province', 'price', 'house_type']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Thiếu một hoặc nhiều cột bắt buộc trong dataset")
        return df
    except Exception as e:
        raise Exception(f"Lỗi khi đọc file CSV: {e}")

# Hàm tiền xử lý và chuyển đổi dữ liệu
def preprocess_data(df, numerical_cols, categorical_cols):
    df = df.copy()
    
    # Điền giá trị thiếu trong cột location
    df['location'] = df['location'].fillna(df['location'].mode()[0])
    
    # Chuẩn hóa tên
    df['location'] = df['location'].str.capitalize()
    df['province'] = df['province'].str.title()
    
    # Mã hóa one-hot
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = one_hot_encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    
    # Mã hóa nhãn
    df['location_label'] = LabelEncoder().fit_transform(df['location'])
    df['province_label'] = LabelEncoder().fit_transform(df['province'])
    df['house_type_label'] = LabelEncoder().fit_transform(df['house_type'])
    
    # Chuẩn hóa min-max
    scaler = MinMaxScaler()
    numerical_scaled = scaler.fit_transform(df[numerical_cols])
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols)
    
    # Chuẩn hóa giá
    y_scaled = scaler.fit_transform(df[['price']]).ravel()
    
    # Kết hợp đặc trưng
    X_one_hot = pd.concat([numerical_scaled_df, encoded_df], axis=1)
    X_label = pd.concat([numerical_scaled_df, df[['location_label', 'province_label', 'house_type_label']]], axis=1)
    X_no_location = numerical_scaled_df
    
    return X_one_hot, X_label, X_no_location, y_scaled, df, one_hot_encoder, scaler

# Hàm tạo biểu đồ trực quan
def generate_visualizations(df, numerical_cols):
    # Biểu đồ nhiệt tương quan
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numerical_cols + ['price']].corr(), annot=True, cmap='coolwarm')
    plt.title('Tương Quan Giữa Các Đặc Trưng')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Phân bố số phòng ngủ
    plt.figure(figsize=(8, 6))
    sns.histplot(df['bedrooms'], kde=True)
    plt.title('Phân Bố Số Phòng Ngủ')
    plt.savefig('bedrooms_distribution.png')
    plt.close()
    
    # Biểu đồ hộp theo loại nhà
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='house_type', y='bedrooms', data=df)
    plt.title('Biểu Đồ Hộp Số Phòng Ngủ Theo Loại Nhà')
    plt.savefig('boxplot_house_types.png')
    plt.close()

# Hàm đánh giá mô hình
def evaluate_models(X_train, X_test, y_train, y_test, models, scenario):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
        print(f"{scenario} - {name}: MSE = {mse:.6f}")
    return results

# Hàm xử lý dữ liệu theo loại nhà
def process_house_type_data(df, house_type, numerical_cols, categorical_cols, one_hot_encoder, scaler):
    df_type = df[df['house_type'] == house_type]
    numerical_scaled = scaler.fit_transform(df_type[numerical_cols])
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols)
    encoded_data = one_hot_encoder.fit_transform(df_type[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    X = pd.concat([numerical_scaled_df, encoded_df], axis=1)
    y = scaler.fit_transform(df_type[['price']]).ravel()
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Định nghĩa các mô hình
models = {
    'LR': LinearRegression(),
    'DT': DecisionTreeRegressor(random_state=42),
    'SVM': SVR(),
    'RF': RandomForestRegressor(random_state=42),
    'XGB': XGBRegressor(random_state=42)
}

# Cấu hình cột
numerical_cols = ['bedrooms', 'land_size', 'living_size']
categorical_cols = ['location', 'province', 'house_type']

# Đọc dữ liệu
df = load_data('house_data.csv')

# Tiền xử lý và chuyển đổi
X_one_hot, X_label, X_no_location, y_scaled, df_processed, one_hot_encoder, scaler = preprocess_data(df, numerical_cols, categorical_cols)

# Tạo biểu đồ trực quan
generate_visualizations(df_processed, numerical_cols)

# Chia dữ liệu
X_train_oh, X_test_oh, y_train, y_test = train_test_split(X_one_hot, y_scaled, test_size=0.2, random_state=42)
X_train_label, X_test_label, _, _ = train_test_split(X_label, y_scaled, test_size=0.2, random_state=42)
X_train_no_loc, X_test_no_loc, _, _ = train_test_split(X_no_location, y_scaled, test_size=0.2, random_state=42)

# Thực nghiệm
# a. Ảnh hưởng của đặc trưng vị trí
print("Thực nghiệm a: Ảnh hưởng của Đặc Trưng Vị Trí")
results_no_location = evaluate_models(X_train_no_loc, X_test_no_loc, y_train, y_test, models, "Không Có Vị Trí")
results_with_location = evaluate_models(X_train_oh, X_test_oh, y_train, y_test, models, "Có Vị Trí")

# b. Mã hóa One-hot so với Mã hóa Nhãn
print("\nThực nghiệm b: Mã hóa One-hot so với Mã hóa Nhãn")
results_one_hot = evaluate_models(X_train_oh, X_test_oh, y_train, y_test, models, "Mã hóa One-hot")
results_label = evaluate_models(X_train_label, X_test_label, y_train, y_test, models, "Mã hóa Nhãn")

# c. Nhà trọ so với Nhà không phải nhà trọ
print("\nThực nghiệm c: Nhà Trọ so với Nhà Không Phải Nhà Trọ")
X_train_nb, X_test_nb, y_train_nb, y_test_nb = process_house_type_data(df_processed, 'Non-Boarding', numerical_cols, categorical_cols, one_hot_encoder, scaler)
X_train_b, X_test_b, y_train_b, y_test_b = process_house_type_data(df_processed, 'Boarding', numerical_cols, categorical_cols, one_hot_encoder, scaler)

results_non_boarding = evaluate_models(X_train_nb, X_test_nb, y_train_nb, y_test_nb, models, "Nhà Không Phải Nhà Trọ")
results_boarding = evaluate_models(X_train_b, X_test_b, y_train_b, y_test_b, models, "Nhà Trọ")

# Lưu kết quả
results_df = pd.DataFrame({
    'Không Có Vị Trí': results_no_location,
    'Có Vị Trí': results_with_location,
    'Mã hóa One-hot': results_one_hot,
    'Mã hóa Nhãn': results_label,
    'Nhà Không Phải Nhà Trọ': results_non_boarding,
    'Nhà Trọ': results_boarding
})
results_df.to_csv('model_results.csv')
print("\nKết quả đã được lưu vào 'model_results.csv'")