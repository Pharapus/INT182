"""เทรนโมเดลแยก spam จากไฟล์ที่ติดป้ายแล้ว (คอลัมน์ text, label)

label ที่ใช้ได้มี 2 ค่า: spam / normal  (ช่องว่างหรือค่าอื่นจะถูกข้ามพร้อมแจ้งเตือน)

วิธีใช้:
    pip install -r requirements.txt
    python train.py                                  # อ่าน data/to_label.csv -> เขียนทับ spam_model.joblib
    python train.py --data my_labels.csv --model out.joblib

ถ้ามี spam_model.joblib เดิมอยู่ จะสำรองเป็น spam_model.joblib.bak ก่อนเขียนทับ
หลังเทรนเสร็จ รีสตาร์ท server (npm start) เพื่อให้ใช้โมเดลใหม่
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from text_utils import MODEL_PATH, ROOT, clean

VALID_LABELS = {"spam", "normal"}
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        # char n-gram เหมาะกับภาษาไทย เพราะไม่ต้องตัดคำ
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                                  min_df=2, sublinear_tf=True, lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0)),
    ])


def load_labeled(csv_path):
    """อ่านไฟล์ที่ติดป้ายแล้ว -> (df, X, y) โดย y=1 คือ spam"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    skipped = df[~df["label"].isin(VALID_LABELS)]
    n_unlabeled = int(skipped["label"].isin({"", "nan"}).sum())
    n_other = len(skipped) - n_unlabeled
    if n_other:
        print(f"คำเตือน: ข้าม {n_other} แถวที่ป้ายไม่ใช่ spam/normal "
              f"(เช่น {sorted(set(skipped['label']) - {'', 'nan'})[:3]})")
    df = df[df["label"].isin(VALID_LABELS)].copy()
    df["text"] = df["text"].map(clean)
    # ตัดข้อความซ้ำ กันโมเดลจำข้อความเดียวกันทั้งฝั่ง train และ test
    df = df.drop_duplicates(subset="text")
    y = (df["label"] == "spam").astype(int)
    return df, df["text"], y


def train(csv_path, model_path) -> None:
    df, X, y = load_labeled(csv_path)
    n_spam, n_ok = int(y.sum()), int((y == 0).sum())
    print(f"ข้อมูลเทรน: spam={n_spam}, ไม่ spam={n_ok}")
    if n_spam < 20 or n_ok < 20:
        print("ตัวอย่างน้อยเกินไป (แนะนำอย่างน้อยประมาณ 50 spam / 50 ไม่ spam) ผลจะไม่น่าเชื่อถือ")
        if min(n_spam, n_ok) < 5:
            sys.exit("ตัวอย่างน้อยกว่า 5 รายการในบางคลาส เทรนไม่ได้")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("\n=== ผลบนชุดทดสอบ (ข้อมูลที่โมเดลไม่เคยเห็น) ===")
    print(classification_report(y_te, pred, target_names=["normal", "spam"], digits=3, zero_division=0))
    cm = confusion_matrix(y_te, pred)
    print("confusion matrix [[TN FP] [FN TP]]:\n", cm)
    report = classification_report(y_te, pred, target_names=["normal", "spam"],
                                   output_dict=True, zero_division=0)

    cv_mean = cv_std = None
    cv_k = min(5, n_spam, n_ok)
    if cv_k >= 2:
        cv = cross_val_score(build_pipeline(), X, y, cv=cv_k, scoring="f1")
        cv_mean, cv_std = float(cv.mean()), float(cv.std())
        print(f"\nF1 (cross-validation, {cv_k} folds): {cv_mean:.3f} ± {cv_std:.3f}")

    # เทรนใหม่ด้วยข้อมูลทั้งหมดก่อนบันทึก
    final = build_pipeline().fit(X, y)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        backup = model_path.with_name(model_path.name + ".bak")
        shutil.copy2(model_path, backup)
        print(f"\nสำรองโมเดลเดิมไว้ที่ {backup}")
    joblib.dump(final, model_path)
    print(f"บันทึกโมเดลที่ {model_path}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": Path(csv_path).name,
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "test_size": int(len(y_te)),
        "test_spam": {k: round(float(report["spam"][k]), 4)
                      for k in ("precision", "recall", "f1-score")},
        "test_confusion_matrix_TN_FP_FN_TP": [int(v) for v in cm.ravel()],
        "cv_f1_mean": None if cv_mean is None else round(cv_mean, 4),
        "cv_f1_std": None if cv_std is None else round(cv_std, 4),
    }
    out = model_path.with_name("metrics.json")
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"บันทึกผลวัดที่ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "to_label.csv"))
    ap.add_argument("--model", default=str(MODEL_PATH))
    a = ap.parse_args()
    train(a.data, a.model)
