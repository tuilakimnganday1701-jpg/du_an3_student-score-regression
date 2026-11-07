from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = PROJECT_ROOT / "data" / "student_scores.csv"
MODEL_PATH: Path = PROJECT_ROOT / "artifacts" / "linear_regression.joblib"

# THỨ TỰ cột đầu vào, phải giống nhau ở lúc huấn luyện và lúc dự đoán
FEATURES: list[str] = [
    "hours_studied", "attendance", "assignments_completed", "previous_score"
]
TARGET: str = "exam_score"  # cột cần dự đoán


def load_and_validate() -> pd.DataFrame:
    """Đọc CSV và kiểm tra đủ cột, không thiếu giá trị."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required].isna().any().any():  # có ô trống nào không?
        raise ValueError("Missing values were found in required columns")
    return df

def main() -> None:
    df = load_and_validate()

    # 80% để học (train), 20% để chấm điểm (test); random_state cố định để lặp lại được
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )

        # Pipeline = chuỗi bước chạy lần lượt: điền thiếu -> chuẩn hóa -> hồi quy
    model: Pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # ô trống -> điền trung vị
        ("scaler", StandardScaler()),                   # đưa mỗi cột về trung bình 0, độ lệch chuẩn 1
        ("regressor", LinearRegression()),              # mô hình hồi quy tuyến tính
    ])

    model.fit(X_train, y_train)  # học CHỈ trên tập train

    prediction = model.predict(X_test)  # dự đoán cho 100 dòng test chưa từng thấy

    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_test, prediction)),         # sai số tuyệt đối trung bình
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),  # căn bình phương sai số
        "r2": float(r2_score(y_test, prediction)),                     # mức độ giải thích dữ liệu
    }
        # Đóng gói mô hình kèm thông tin cần để dùng lại đúng cách
    artifact: dict[str, Any] = {
        "model": model,                       # cả Pipeline đã huấn luyện
        "feature_names": FEATURES,            # thứ tự cột
        "target_name": TARGET,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,  # để phát hiện lệch phiên bản
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)  # nén mức 3 cho file nhỏ
    print(f"Saved model to {MODEL_PATH}")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
