"""เทรนโมเดลแยก spam จาก data/to_label.csv (คอลัมน์ text, label)

โมเดลเรียนรู้จาก 2 อย่างพร้อมกัน: เนื้อหาข้อความ (char n-gram) และ "จำนวนครั้งที่ข้อความซ้ำ"
(นับต่อวิดีโอจาก data/comments.jsonl ด้วย tools/run_dups.mjs ซึ่งเป็นโค้ดชุดเดียวกับเว็บ)
ไม่มีกฎหรือ threshold ของการซ้ำนอกโมเดล ว่าซ้ำแบบไหนคือ spam ขึ้นกับป้ายที่ติดไว้

label ที่ใช้ได้มี 2 ค่า: spam / normal  (ช่องว่างหรือค่าอื่น เช่น low_quality จะถูกข้ามพร้อมแจ้งเตือน)
ถ้ามี data/dup_to_label.csv (สร้างด้วย tools/make_dup_to_label.py) ป้ายในนั้นจะทับป้ายใน to_label.csv
เฉพาะแถวที่เติมแล้ว

วิธีใช้:
    python ml/train.py
    python ml/train.py --data my_labels.csv --model models/my.joblib
    python ml/train.py --no-dup      # ไม่ใช้ข้อมูลข้อความซ้ำ (ทุกข้อความนับว่าไม่ซ้ำ)

หลักการติดป้าย: อีโมจิ/เครื่องหมายล้วน หรือข้อความสั้น <= 3 ตัวอักษร = spam
               ข้อความที่ยาวกว่านั้นและมีตัวอักษร = normal (เว้นแต่เป็นโฆษณา/ชวนติดต่อจริง = spam)
               ข้อความที่ซ้ำ: spam ถ้าถูกโพสต์ซ้ำแบบโปรโมต/ก็อปวาง/บอท, normal ถ้าบังเอิญตรงกัน (เช่น "ขอบคุณครับ")
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import MODEL_PATH, ROOT, clean, dup_counts, dup_features  # noqa: E402

VALID_LABELS = {"spam", "normal"}
RANDOM_STATE = 42
COMMENTS_PATH = ROOT / "data" / "comments.jsonl"
OVERRIDES_PATH = ROOT / "data" / "dup_to_label.csv"


def build_pipeline(use_dup: bool = True) -> Pipeline:
    """อินพุตเป็น DataFrame (คอลัมน์ text, dup) use_dup=False = ใช้แค่ข้อความ (ไว้เทียบใน evaluate.py)"""
    parts = [
        # char n-gram เหมาะกับภาษาไทย เพราะไม่ต้องตัดคำ
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                                  min_df=2, sublinear_tf=True, lowercase=True), "text"),
    ]
    if use_dup:
        parts.append(("dup", FunctionTransformer(dup_features, validate=False), ["dup"]))
    return Pipeline([
        ("features", ColumnTransformer(parts)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0)),
    ])


def load_overrides(path):
    """อ่าน data/dup_to_label.csv -> {ข้อความ: ป้าย} เฉพาะแถวที่เติมป้ายเป็น spam/normal แล้ว"""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if not {"text", "label"} <= set(df.columns):
        sys.exit(f"{path} ต้องมีคอลัมน์ text และ label")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)]
    return {clean(t): l for t, l in zip(df["text"], df["label"])}


def load_dup_map(comments_path):
    """อ่านคอมเมนต์ดิบ -> {ข้อความ: จำนวนซ้ำสูงสุดต่อวิดีโอ} (นับต่อวิดีโอเหมือนที่เว็บนับ)"""
    if comments_path is None or not Path(comments_path).exists():
        return {}
    rows = [json.loads(l) for l in open(comments_path, encoding="utf-8") if l.strip()]
    mp = {}
    for r, c in zip(rows, dup_counts(rows)):
        t = clean(r.get("text", ""))
        mp[t] = max(mp.get(t, 1), c)
    return mp


def load_labeled(csv_path, overrides_path=None, comments_path=None):
    """อ่านไฟล์ที่ติดป้ายแล้ว -> (df, X, y) โดย X = DataFrame(text, dup), y=1 คือ spam
    comments_path=None หมายถึงไม่มีข้อมูลซ้ำ (ทุกแถว dup=1)"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    skipped = df[~df["label"].isin(VALID_LABELS)]
    n_unlabeled = int((skipped["label"].isin({"", "nan"})).sum())
    n_other = len(skipped) - n_unlabeled
    if n_other:
        print(f"คำเตือน: ข้าม {n_other} แถวที่ป้ายไม่ใช่ spam/normal "
              f"(เช่น {sorted(set(skipped['label']) - {'', 'nan'})[:3]}) ให้แก้ป้ายเป็น spam หรือ normal")
    df = df[df["label"].isin(VALID_LABELS)].copy()
    df["text"] = df["text"].map(clean)

    overrides = load_overrides(overrides_path)
    if overrides:
        new = df["text"].map(overrides)
        changed = int((new.notna() & (new != df["label"])).sum())
        df["label"] = new.fillna(df["label"])
        print(f"ใช้ป้ายจาก dup_to_label.csv {len(overrides)} รายการ (เปลี่ยนป้ายเดิม {changed} แถว)")

    # ตัดข้อความซ้ำ กันโมเดลจำข้อความเดียวกันทั้งฝั่ง train และ test (จำนวนซ้ำเก็บไว้ในคอลัมน์ dup)
    df = df.drop_duplicates(subset="text").copy()
    dup_map = load_dup_map(comments_path)
    df["dup"] = df["text"].map(dup_map).fillna(1).astype(int)
    if dup_map:
        missing = int((~df["text"].isin(dup_map)).sum())
        if missing:
            print(f"หมายเหตุ: {missing} แถวไม่พบใน {Path(comments_path).name} นับเป็นไม่ซ้ำ (dup=1)")
    y = (df["label"] == "spam").astype(int)
    return df, df[["text", "dup"]], y


def train(csv_path, model_path, overrides_path=None, comments_path=None, use_dup=True) -> None:
    df, X, y = load_labeled(csv_path, overrides_path, comments_path)
    n_spam, n_ok = int(y.sum()), int((y == 0).sum())
    print(f"ข้อมูลเทรน: spam={n_spam}, ไม่ spam={n_ok}  ")
    print("จำนวนตามป้ายเดิม:", df["label"].value_counts().to_dict())
    is_dup = df["dup"] >= 2
    n_dup = int(is_dup.sum())
    n_dup_spam = int((is_dup & (y == 1)).sum())
    print(f"ข้อความที่ซ้ำ (>= 2 ครั้ง): {n_dup} แถว เป็น spam {n_dup_spam} แถว")
    if use_dup and n_dup_spam < 10:
        print("คำเตือน: ข้อความซ้ำที่ติดป้าย spam มีน้อยกว่า 10 แถว โมเดลเรียนรู้เรื่องการซ้ำไม่ได้จริง "
              "ให้ติดป้ายใน data/dup_to_label.csv เพิ่ม (ดู README)")
    if n_spam < 20 or n_ok < 20:
        print("ตัวอย่างน้อยเกินไป (แนะนำอย่างน้อยประมาณ 50 spam / 50 ไม่ spam) ผลจะไม่น่าเชื่อถือ")
        if min(n_spam, n_ok) < 5:
            sys.exit("ตัวอย่างน้อยกว่า 5 รายการในบางคลาส เทรนไม่ได้")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = build_pipeline(use_dup).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("\n=== ผลบนชุดทดสอบ (ข้อมูลที่โมเดลไม่เคยเห็น) ===")
    print(classification_report(y_te, pred, target_names=["normal", "spam"], digits=3))
    cm = confusion_matrix(y_te, pred)
    print("confusion matrix [[TN FP] [FN TP]]:\n", cm)
    report = classification_report(y_te, pred, target_names=["normal", "spam"],
                                   output_dict=True, zero_division=0)

    cv_mean = cv_std = None
    cv_k = min(5, n_spam, n_ok)
    if cv_k >= 2:
        cv = cross_val_score(build_pipeline(use_dup), X, y, cv=cv_k, scoring="f1")
        cv_mean, cv_std = float(cv.mean()), float(cv.std())
        print(f"\nF1 (cross-validation, {cv_k} folds): {cv_mean:.3f} ± {cv_std:.3f}")

    # เทรนใหม่ด้วยข้อมูลทั้งหมดก่อนบันทึก
    final = build_pipeline(use_dup).fit(X, y)
    dup_coef = None
    if use_dup:
        w = final.named_steps["clf"].coef_[0][-2:]
        dup_coef = {"is_dup": round(float(w[0]), 3), "log_count": round(float(w[1]), 3)}
        print(f"น้ำหนักของ feature ข้อความซ้ำ: {dup_coef}  (บวก = ซ้ำแล้วเสี่ยง spam, ลบ = ซ้ำแล้วดูเป็นปกติ)")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final, model_path)
    print(f"\nบันทึกโมเดลที่ {model_path}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": str(Path(csv_path).name),
        "labels": ["normal", "spam"],
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "n_positive": n_spam,
        "n_negative": n_ok,
        "uses_dup_feature": bool(use_dup),
        "n_dup_rows": n_dup,
        "n_dup_spam_rows": n_dup_spam,
        "dup_coefficients": dup_coef,
        "test_size": int(len(y_te)),
        "test_spam": {k: round(float(report["spam"][k]), 4)
                      for k in ("precision", "recall", "f1-score")},
        "test_confusion_matrix_TN_FP_FN_TP": [int(v) for v in cm.ravel()],
        "cv_f1_mean": None if cv_mean is None else round(cv_mean, 4),
        "cv_f1_std": None if cv_std is None else round(cv_std, 4),
        "note": "ข้อมูล spam จริงมีน้อย ตัวเลขนี้ไม่ควรตีความเกินกว่าชุดข้อมูลนี้",
    }
    out = model_path.with_name("metrics.json")
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"บันทึกผลวัดที่ {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "to_label.csv"))
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--comments", default=str(COMMENTS_PATH), help="คอมเมนต์ดิบ ใช้นับข้อความซ้ำ")
    ap.add_argument("--overrides", default=str(OVERRIDES_PATH), help="ป้ายของข้อความซ้ำ (ถ้าไฟล์มีอยู่)")
    ap.add_argument("--no-dup", action="store_true", help="ไม่ใช้ข้อมูลข้อความซ้ำ")
    a = ap.parse_args()
    train(a.data, a.model, a.overrides, None if a.no_dup else a.comments, use_dup=not a.no_dup)
