import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

def display_dataframe_info(df):
    info_df = pd.DataFrame(columns=['Cột', 'Kiểu dữ liệu', 'Số giá trị thiếu', '% Thiếu', 'Số giá trị duy nhất', '% Giá trị duy nhất'])
    for col in df.columns:
        dtype = df[col].dtype
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        unique_percent = (unique_count / len(df)) * 100
        info_df.loc[len(info_df)] = [col, dtype, missing_count, f"{missing_percent:.2f}%", unique_count, f"{unique_percent:.2f}%"]
    return info_df

# Nhận biết những cột không có ý nghĩa, để xóa
def detect_useless_columns(data, uniqueness_thresh=0.75):
    drop_cols = []
    n_rows = len(data)
    
    for col in data.columns:
        nunique = data[col].nunique()
        ratio_unique = nunique / n_rows
        
        if data[col].dtype in ['object', 'category'] and ratio_unique > uniqueness_thresh:
                drop_cols.append(col)
        #elif any(key in col.lower() for key in ['id', 'name']):
        elif col.lower() in ['id', 'name', 'passengerid']:
            drop_cols.append(col)
    
    return drop_cols

def auto_clean(data, target_col):
    # 1. Chuẩn hóa tên cột
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

    # 2. Loại bỏ cột trống hoàn toàn
    data = data.dropna(axis=1, how='all')

    # 3. Loại bỏ dòng trống hoàn toàn hoặc có số mẫu ít hơn 10
    data = data.dropna(axis=0, how='all')
    value_counts = data[target_col].value_counts()
    valid_classes = value_counts[value_counts >= 3].index
    data = data[data[target_col].isin(valid_classes)].copy()

    # 4. Loại bỏ trùng lặp
    data = data.drop_duplicates()

    # 5. Làm sạch chuỗi
    for col in data.select_dtypes(include='object'):
        data[col] = data[col].str.strip()

    # 6. Xóa cột không cần thiết
    drop_cols = detect_useless_columns(data)
    data = data.drop(columns=drop_cols)

    # 7. Phân chia X và y nếu có
    has_target = target_col in data.columns
    features_df = data.drop(columns=[target_col]) if has_target else data.copy()

    # 8. Xác định cột phân loại và số
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Nếu cột numeric có ít unique → cũng coi là categorical
    for col in features_df.columns:
        if features_df[col].nunique() < 10 and features_df[col].dtype != 'object':
            if col not in categorical_cols:
                categorical_cols.append(col)
            if col in numerical_cols:
                numerical_cols.remove(col)

    # 9. Encode categorical
    for col in categorical_cols:
        features_df[col] = LabelEncoder().fit_transform(features_df[col])

    # 11. Thêm lại cột target nếu có
    if has_target:
        cleaned_df = features_df.copy()
        y = data[target_col]

        # Nếu y không phải số nguyên, hoặc không liên tiếp từ 0 → encode
        if y.dtype == 'object' or y.dtype.name == 'category' or not np.issubdtype(y.dtype, np.integer):
            le = LabelEncoder()
            y = le.fit_transform(y)
            cleaned_df[target_col] = y
        else:
            cleaned_df[target_col] = y
    else:
        return features_df
    
    return cleaned_df

def safe_impute(X, strategy="median"):
    X = X.copy()
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy=strategy)
    return imp.fit_transform(X)

def estimate_noise_level_from_outliers(X):
    X_filled = safe_impute(X, strategy='median')
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(X_filled)
    outlier_ratio = np.mean(preds == -1)
    return np.clip(outlier_ratio * 2.0, 0.01, 1.0)

def estimate_noise_level_from_variance(X):
    X_filled = np.nan_to_num(X, nan=0.0)  # hoặc median nếu bạn muốn chính xác hơn
    stds = np.std(X_filled, axis=0)
    means = np.mean(np.abs(X_filled), axis=0) + 1e-6
    relative_noise = stds / means
    return np.clip(relative_noise.mean(), 0.01, 1.0)

def estimate_noise_level(X):
    var_score = estimate_noise_level_from_variance(X)
    outlier_score = estimate_noise_level_from_outliers(X)
    return np.clip((var_score + outlier_score) / 2, 0.01, 1.0)