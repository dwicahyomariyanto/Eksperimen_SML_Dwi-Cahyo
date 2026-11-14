"""
automate_Dwi-Cahyo.py
======================
Otomatisasi preprocessing dataset Credit Scoring (German Credit).

Tahapan di file ini merupakan konversi langsung dari notebook
`Eksperimen_Dwi-Cahyo.ipynb` (bagian 5. Data Preprocessing):

1. Memuat data mentah
2. Menghapus data duplikat & menangani missing value
3. Encoding target  (1 = good -> 0, 2 = bad/default -> 1)
4. Menangani outlier dengan IQR capping pada fitur numerik
5. Split data train/test (stratified, 80:20)
6. Standarisasi fitur numerik & One-Hot Encoding fitur kategorikal
   (fit hanya pada data train untuk mencegah data leakage)
7. Menyimpan data yang siap dilatih ke folder `credit_preprocessing`

Contoh penggunaan:
    python preprocessing/automate_Dwi-Cahyo.py \
        --input credit_raw/credit_raw.csv \
        --output-dir preprocessing/credit_preprocessing
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default"
NUMERIC_FEATURES = [
    "months_loan_duration",
    "amount",
    "installment_rate",
    "residence_history",
    "age",
    "existing_credits",
    "dependents",
]
OUTLIER_FEATURES = ["months_loan_duration", "amount", "age"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    df = pd.read_csv(path)
    print(f"[load] {path} -> shape {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus duplikat dan menangani missing value."""
    df = df.drop_duplicates().copy()
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    print(f"[clean] shape setelah cleaning {df.shape}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah target: 1 (good) -> 0, 2 (bad/default) -> 1."""
    df = df.copy()
    df[TARGET] = df[TARGET].map({1: 0, 2: 1}).astype(int)
    return df


def cap_outliers(df: pd.DataFrame, columns=OUTLIER_FEATURES) -> pd.DataFrame:
    """Capping outlier dengan metode IQR (batas Q1-1.5*IQR dan Q3+1.5*IQR)."""
    df = df.copy()
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def split_data(df: pd.DataFrame):
    """Split data menjadi train & test secara stratified."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Membuat ColumnTransformer: StandardScaler + OneHotEncoder."""
    categorical = [c for c in X.columns if c not in NUMERIC_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def preprocess_data(input_path: str, output_dir: str):
    """Pipeline preprocessing lengkap. Mengembalikan (train_df, test_df)."""
    df = load_data(input_path)
    df = clean_data(df)
    df = encode_target(df)
    df = cap_outliers(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    train_df = pd.DataFrame(X_train_t, columns=feature_names)
    train_df[TARGET] = y_train.to_numpy()
    test_df = pd.DataFrame(X_test_t, columns=feature_names)
    test_df[TARGET] = y_test.to_numpy()

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    print(f"[save] train {train_df.shape}, test {test_df.shape} -> {output_dir}")
    return train_df, test_df


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Preprocessing Credit Scoring")
    parser.add_argument(
        "--input",
        default=os.path.join(base, "..", "credit_raw", "credit_raw.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(base, "credit_preprocessing"),
    )
    args = parser.parse_args()
    preprocess_data(args.input, args.output_dir)


if __name__ == "__main__":
    main()
