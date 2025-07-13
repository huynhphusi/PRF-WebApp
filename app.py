import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import KFold   # Dùng để tạo py
from PRF import prf

from utils import display_dataframe_info, auto_clean, estimate_noise_level

st.set_page_config(page_title="Ứng dụng ML", layout="wide")
st.title("🤖 Ứng dụng ML trực quan từ CSV")
st.markdown("""
    ### Hướng dẫn sử dụng
    1. Tải lên file CSV chứa dữ liệu của bạn.
    2. Chọn cột mục tiêu (target) để huấn luyện mô hình.
    3. Lựa chọn mô hình và nhấn "Train mô hình" để bắt đầu huấn luyện.
    4. Tải lên file CSV mới để dự đoán kết quả.
""")
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Upload CSV
ordinary_file = st.file_uploader("📁 Tải lên file CSV", type=["csv"])

if ordinary_file:
    # Check if a new file is uploaded or if the file has changed
    if "ordinary_filename" not in st.session_state or ordinary_file.name != st.session_state["ordinary_filename"]:
        df = pd.read_csv(ordinary_file)
        if len(df) > 100000:
            st.warning("Dữ liệu lớn (>100,000 dòng). Ứng dụng sẽ lấy mẫu ngẫu nhiên 100,000 dòng để xử lý.")
            df = df.sample(n=100000, random_state=42)
        st.session_state["ordinary_df"] = df
        st.session_state["ordinary_filename"] = ordinary_file.name
        # Clear previous training results when a new dataset is uploaded
        if "trained_model" in st.session_state:
            del st.session_state["trained_model"]
            del st.session_state["X_test"]
            del st.session_state["y_test"]
            del st.session_state["X_train"]
            del st.session_state["y_train"]
            del st.session_state["model_name"]
            del st.session_state["feature_names"]
            del st.session_state["col_target"]

# Lấy dữ liệu từ session_state để dùng
if "ordinary_df" in st.session_state:
    df = st.session_state["ordinary_df"]
else:
    st.warning("⚠️ Vui lòng tải lên một file CSV để bắt đầu.")
    st.stop()

st.subheader("📊 Dữ liệu đã tải lên:")
st.dataframe(df.head())

st.subheader("📌 Thống kê dữ liệu số:")
st.write(df.describe())

st.subheader("📌 Thống kê kiểu dữ liệu:")

st.dataframe(display_dataframe_info(df))

# Biểu đồ tròn (dùng cả cột số rời rạc)
st.subheader("🥧 Biểu đồ tròn (cho cột rời rạc):")

# Chọn các cột có < 20 giá trị duy nhất
discrete_cols = [col for col in df.columns if df[col].nunique() < 20]

if discrete_cols:
    pie_col = st.selectbox("Chọn cột rời rạc:", discrete_cols, key="pie")
    pie_data = df[pie_col].value_counts().reset_index()
    pie_data.columns = [pie_col, 'count']
    fig_pie = px.pie(pie_data, values='count', names=pie_col, title=f"Phân bố giá trị trong '{pie_col}'")
    st.plotly_chart(fig_pie)
else:
    st.info("Không có cột rời rạc để vẽ biểu đồ tròn.")

st.subheader("📊 Làm sạch dữ liệu:")
col_target = st.selectbox("Cột mục tiêu (target):", df.columns)
if col_target and col_target != df.columns[0]:
    col_target = col_target.strip().lower().replace(' ', '_')
    df_prf = auto_clean(df, target_col=col_target)

    df_cleaned = df_prf.copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df_cleaned.drop(columns=col_target))
    st.session_state['imputer'] = imputer # Lưu lại imputer
    df_noprf = pd.DataFrame(X_imputed, columns=df_cleaned.columns[:-1])
    df_noprf[col_target] = df_cleaned[col_target].values

    st.session_state["df_prf"] = df_prf
    st.session_state["df_noprf"] = df_noprf
    st.session_state["col_target"] = col_target

if "col_target" in st.session_state:
    col_target = st.session_state["col_target"]
if "df_prf" in st.session_state:
    df_prf = st.session_state["df_prf"]
if "df_noprf" in st.session_state:
    df_noprf = st.session_state["df_noprf"]

    st.dataframe(df_noprf.head())

    st.subheader("🧠 Huấn luyện dữ liệu:")
    model_name = st.selectbox("🧠 Chọn mô hình:", ["Random Forest", "SVM", "XGBoost", "Probabilistic Random Forest"])

    # Train mô hình button
    if st.button("🚀 Train mô hình"):
        X = df_noprf.drop(columns=[col_target])
        y = df_noprf[col_target]
        
        # Kiểm tra nếu X và y có số lượng mẫu phù hợp
        if len(X) != len(y):
            st.error("Số lượng hàng trong features (X) và target (y) không khớp. Vui lòng kiểm tra lại dữ liệu CSV và cột mục tiêu.")
            st.stop()
        
        # Kiểm tra nếu y có nhiều hơn 1 lớp cho phân loại
        if y.nunique() < 2:
            st.error("Cột mục tiêu chỉ có một hoặc không có giá trị duy nhất. Không thể huấn luyện mô hình phân loại.")
            st.stop()


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=42)
        elif model_name == "SVM":
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        elif model_name == "XGBoost":
            model = XGBClassifier(n_estimators=100, max_depth=8, min_child_weight=10, learning_rate=0.1, eval_metric='mlogloss', random_state=42)
        else:
            X_prf = df_prf.drop(columns=[col_target])  # Drop the target column for features
            y_prf = df_prf[col_target]
            X_prf = X_prf.values if isinstance(X_prf, pd.DataFrame) else X_prf
            y_prf = y_prf.values if isinstance(y_prf, pd.Series) else y_prf
            noise_level = estimate_noise_level(X_prf)
            dX = noise_level * np.abs(np.nan_to_num(X_prf, nan=1.0)) + 0.01
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            n_classes = len(np.unique(y_prf))
            py = np.zeros((len(X_prf), n_classes))

            for train_idx, val_idx in kf.split(X_prf):
                rf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=41)
                rf.fit(X_prf[train_idx], y_prf[train_idx])
                py[val_idx] = rf.predict_proba(X_prf[val_idx])
            X_train, X_test, dX_train, dX_test, py_train, py_test = train_test_split(
                X_prf, dX, py, test_size=0.2, random_state=40, stratify=np.argmax(py, axis=1)
            )
            model = prf(n_estimators=100, bootstrap=True, max_depth=8, keep_proba=1.0, n_jobs=-1)

        try:
            with st.spinner("⏳ Đang huấn luyện mô hình, vui lòng chờ..."):
                if model_name == "Probabilistic Random Forest":
                    model.fit(X=X_train, dX=dX_train, py=py_train)
                    st.session_state["noise_level"] = noise_level
                    st.session_state["dX_test"] = dX_test
                    st.session_state["py_test"] = py_test
                    st.session_state["dX_train"] = dX_train
                    st.session_state["py_train"] = py_train
                else:
                    model.fit(X_train, y_train)
                
            # ✅ Lưu model & dữ liệu vào session_state
            st.session_state["trained_model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["model_name"] = model_name
            st.session_state["feature_names"] = X.columns.tolist() # Convert to list for session_state compatibility
            st.session_state["col_target"] = col_target

            st.success(f"✅ Mô hình **{model_name}** đã được huấn luyện!")
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {e}")

    # --- Hiển thị kết quả huấn luyện (chỉ khi có mô hình đã huấn luyện) ---
    if "trained_model" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Kết quả huấn luyện mô hình")

        model = st.session_state["trained_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        model_name = st.session_state["model_name"]
        col_target = st.session_state["col_target"]
        feature_names = st.session_state["feature_names"]

        with st.spinner("⏳ Đang đánh giá kết quả huấn luyện, vui lòng chờ..."):
            if model_name == "Probabilistic Random Forest":
                noise_level = st.session_state["noise_level"]
                dX_test = st.session_state["dX_test"]
                py_test = st.session_state["py_test"]

                y_test = py_test.argmax(axis=1)
                y_proba_prf = np.array(model.predict_proba(X_test, dX=dX_test))
                y_pred = y_proba_prf.argmax(axis=1)
            else:
                y_pred = model.predict(X_test)

            st.text("📄 Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            st.text("📉 Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=list(map(str, range(cm.shape[1]))), y=list(map(str, range(cm.shape[0]))),
                annotation_text=cm, colorscale='Blues'
            )
            fig_cm.update_layout(xaxis_title="Dự đoán", yaxis_title="Thực tế")
            st.plotly_chart(fig_cm)
        
        # --- Phần dự đoán dữ liệu mới (chỉ khi có mô hình đã huấn luyện) ---
        st.markdown("---")
        st.header("🚀 Dự đoán với dữ liệu mới")

        st.subheader("📥 Tải lên CSV mới để dự đoán")
        predict_file = st.file_uploader("Tải lên file CSV mới để dự đoán", type=["csv"], key="predict")

        if predict_file:
            df_pre = pd.read_csv(predict_file)
            st.dataframe(df_pre)
            
            df_pre_prf = auto_clean(df_pre, target_col=col_target)

            df_cleaned = df_pre_prf.copy()
            if 'imputer' in st.session_state:
                imputer = st.session_state['imputer']
            X_imputed = imputer.transform(df_cleaned.drop(columns=col_target))
            df_pre_noprf = pd.DataFrame(X_imputed, columns=df_cleaned.columns[:-1])
            df_pre_noprf[col_target] = df_cleaned[col_target].values

            st.subheader("📊 Dữ liệu đã làm sạch:")
            st.dataframe(df_pre_noprf.head())

            # Kiểm tra xem các cột của dữ liệu mới có khớp với các cột đã huấn luyện không
            if not all(col in df_pre_noprf.columns[:-1].tolist() for col in feature_names):
                st.error("Tệp CSV mới không chứa tất cả các cột tính năng được sử dụng để huấn luyện mô hình. Vui lòng đảm bảo các cột khớp.")
            else:
                try:
                    if model_name == "Probabilistic Random Forest":
                        X_pre = df_pre_prf[feature_names]
                        X_pre = X_pre.values if isinstance(X_pre, pd.DataFrame) else X_pre
                        predictions = np.array(model.predict_proba(X_pre, dX=np.abs(X_pre) * noise_level + 0.01)).argmax(axis=1)
                    else:
                        X_pre = df_pre_noprf[feature_names]
                        predictions = model.predict(X_pre)
                    df_pre_noprf["Dự đoán"] = predictions
                    
                    # If the model has predict_proba (like Logistic Regression, Random Forest, but not always SVM)
                    if hasattr(model, 'predict_proba'):
                        probabilities = np.array(model.predict_proba(X_pre))
                        # Get class labels
                        if model_name == "Probabilistic Random Forest":
                            class_labels = list(range(probabilities.shape[1]))  # Gán nhãn là 0,1,...n_classes-1
                        else:
                            class_labels = model.classes_
                        # Add probability columns
                        for i, label in enumerate(class_labels):
                            df_pre_noprf[f"Xác suất {label}"] = probabilities[:, i]

                    st.subheader("📋 Kết quả dự đoán:")
                    st.dataframe(df_pre_noprf)
                    
                    csv = df_pre_noprf.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Tải kết quả CSV", data=csv, file_name="du_doan.csv", mime='text/csv')
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}. Vui lòng kiểm tra định dạng dữ liệu và đảm bảo các cột số khớp với dữ liệu huấn luyện.")
    else:
        st.info("💡 Vui lòng huấn luyện mô hình trước khi dự đoán dữ liệu mới.")