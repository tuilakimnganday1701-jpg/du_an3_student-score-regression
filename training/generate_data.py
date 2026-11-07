from pathlib import Path
import numpy as np
import pandas as pd

# Thư mục gốc dự án = lùi 1 cấp từ thư mục chứa file này (training/ -> gốc)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "student_scores.csv"


def main(n_samples: int = 500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)  # bộ sinh số ngẫu nhiên, cố định seed để tái lập

    # 4 đặc trưng đầu vào (feature)
    hours = rng.uniform(0, 20, n_samples)           # giờ học: 0-20
    attendance = rng.uniform(50, 100, n_samples)    # chuyên cần: 50-100%
    assignments = rng.integers(0, 11, n_samples)    # số bài tập: 0-10 (11 không được lấy)
    previous = rng.uniform(0, 10, n_samples)        # điểm trước: 0-10
    noise = rng.normal(0, 0.55, n_samples)          # nhiễu ngẫu nhiên, độ lệch chuẩn 0.55

    # "Quy luật thật" của dữ liệu + nhiễu, rồi ép điểm về thang 0-10
    exam = np.clip(
        0.12 * hours + 0.025 * attendance + 0.18 * assignments
        + 0.42 * previous - 2 + noise,
        0,
        10,
    )

    # Gom thành bảng, làm tròn 2 chữ số
    frame = pd.DataFrame({
        "hours_studied": hours.round(2),
        "attendance": attendance.round(2),
        "assignments_completed": assignments,
        "previous_score": previous.round(2),
        "exam_score": exam.round(2),  # cột cần dự đoán (target)
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # tạo thư mục data/ nếu chưa có
    frame.to_csv(OUTPUT_PATH, index=False)                 # ghi CSV, không kèm cột số thứ tự
    print(f"Created {OUTPUT_PATH} with {len(frame)} rows")


if __name__ == "__main__":
    main()