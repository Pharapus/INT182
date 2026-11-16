"""เทรนโมเดลแยก spam จากไฟล์ที่ติดป้ายแล้ว (คอลัมน์ text, label)

label ที่ใช้ได้มี 2 ค่า: spam / normal  (ช่องว่างหรือค่าอื่นจะถูกข้ามพร้อมแจ้งเตือน)

โมเดลดู 2 อย่าง: (1) เนื้อความ  (2) พฤติกรรมการโพสต์ซ้ำ (dup, author_dup)
ไฟล์ CSV ไม่มีข้อมูลบัญชี/การซ้ำ ถ้าไม่มีคอลัมน์ dup, author_dup ในไฟล์
ตัวเทรนจะ "จำลอง" ข้อมูลส่วนนี้ขึ้นมา (ดู augment()) และเติมแถวตัวอย่างการโพสต์ซ้ำเข้าไป
**ค่าที่โมเดลเรียนรู้เรื่องการซ้ำจึงมาจากสมมติฐานในโค้ด ไม่ใช่ข้อมูลจริง**
ถ้ามีคอลัมน์ dup, author_dup จริงใน CSV จะใช้ค่าจริงและไม่จำลอง

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
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from text_utils import MODEL_PATH, ROOT, SpamModel, clean, meta_features

VALID_LABELS = {"spam", "normal"}
RANDOM_STATE = 42

# ---- สมมติฐานของข้อมูลจำลอง (ปรับได้) ----
N_REPEATER_SPAM = 300     # แถวใหม่: บัญชีเดียวโพสต์ข้อความเดิมซ้ำหลายครั้ง -> spam
N_SHARED_NORMAL = 200     # แถวใหม่: ข้อความสั้นที่หลายบัญชีพิมพ์ตรงกัน (555 ฯลฯ) -> normal
SHORT_LEN = 30            # ถือว่าข้อความ "สั้น" เมื่อยาวไม่เกินนี้


def build_text_pipeline() -> Pipeline:
    """ชั้น 1: ดูเฉพาะเนื้อความ (เหมือนโมเดลเดิม)"""
    return Pipeline([
        # char n-gram เหมาะกับภาษาไทย เพราะไม่ต้องตัดคำ
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                                  min_df=2, sublinear_tf=True, lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0)),
    ])


def build_meta() -> LogisticRegression:
    """ชั้น 2: รวมคะแนนเนื้อความ + พฤติกรรมซ้ำ"""
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_spam_model(df, y, rng, has_real):
    """เทรนทั้ง 2 ชั้นจาก df (ต้องมีคอลัมน์ text, label) -> SpamModel"""
    texts = df["text"].tolist()
    k = min(5, int(y.sum()), int((y == 0).sum()))
    # คะแนนเนื้อความแบบ out-of-fold: ชั้น 2 จะได้ไม่เห็นคะแนนที่ชั้น 1 "จำเฉลย" ไว้แล้ว
    oof = cross_val_predict(build_text_pipeline(), texts, y, cv=max(k, 2),
                            method="predict_proba")[:, 1]
    base = df[["text", "label"]].copy()
    base["tlogit"] = logit(oof)
    if has_real:
        base["dup"], base["author_dup"] = df["dup"], df["author_dup"]
    else:
        base = simulate_repeats(base, y, rng)
        base["tlogit"] = logit(oof)
        full = pd.concat([base, synthetic_rows(base, rng)], ignore_index=True)
        base = full
    X = meta_features(base["tlogit"], base["dup"], base["author_dup"])
    meta = build_meta().fit(X, (base["label"] == "spam").astype(int))
    text_model = build_text_pipeline().fit(texts, y)
    return SpamModel(text_model, meta)


def load_labeled(csv_path):
    """อ่านไฟล์ที่ติดป้ายแล้ว -> (df, y) โดย y=1 คือ spam"""
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
    return df, y


def simulate_repeats(df, y, rng):
    """ใส่ dup / author_dup จำลองให้แถวจริง (ใช้เมื่อ CSV ไม่มีข้อมูลนี้)

    สมมติฐาน: คอมเมนต์ปกติแทบไม่ซ้ำ; spam ส่วนหนึ่งถูกโพสต์ซ้ำโดยบัญชีเดิม
    """
    n = len(df)
    author_dup = np.ones(n, dtype=int)
    dup = np.ones(n, dtype=int)
    is_spam = y.to_numpy() == 1

    # normal: 3% ที่บัญชีเดียวพิมพ์ซ้ำ 2 ครั้ง, 10% ที่มีคนอื่นพิมพ์ตรงกัน
    r = rng.random(n)
    author_dup[~is_spam & (r < 0.03)] = 2
    dup[~is_spam & (r < 0.10)] = rng.integers(2, 5, size=int((~is_spam & (r < 0.10)).sum()))
    # spam: ครึ่งหนึ่งถูกโพสต์ซ้ำโดยบัญชีเดิม 2-6 ครั้ง
    rep = is_spam & (rng.random(n) < 0.5)
    author_dup[rep] = rng.integers(2, 7, size=int(rep.sum()))
    dup[is_spam & ~rep] = rng.integers(1, 4, size=int((is_spam & ~rep).sum()))
    dup = np.maximum(dup, author_dup)

    out = df.copy()
    out["dup"], out["author_dup"] = dup, author_dup
    return out


def synthetic_rows(df, rng):
    """แถวสังเคราะห์ที่สอนกฎของพฤติกรรม ใช้ข้อความจริงจากชุด normal

    - บัญชีเดียวโพสต์ข้อความเดิม >= 3 ครั้ง -> spam (แม้เนื้อความดูปกติ/สั้น เช่น "testing")
    - ข้อความสั้นที่หลายบัญชีพิมพ์ตรงกัน แต่ไม่มีใครซ้ำเอง -> normal
    """
    normal = df[df["label"] == "normal"]
    short = normal[normal["text"].str.len() <= SHORT_LEN]
    pool = short if len(short) >= 20 else normal

    def pick(n, prefer_short):
        src = pool if prefer_short else normal
        return src.sample(n, replace=len(src) < n, random_state=int(rng.integers(1_000_000)))

    # ครึ่งหนึ่งเน้นข้อความสั้น (เคส testing) อีกครึ่งใช้ข้อความปกติทั่วไป
    rep = pd.concat([pick(N_REPEATER_SPAM // 2, True), pick(N_REPEATER_SPAM - N_REPEATER_SPAM // 2, False)])
    rep = rep[["text", "tlogit"]].copy()
    rep["label"] = "spam"
    rep["author_dup"] = rng.integers(3, 9, size=len(rep))
    rep["dup"] = rep["author_dup"] + rng.integers(0, 3, size=len(rep))

    shared = pick(N_SHARED_NORMAL, True)[["text", "tlogit"]].copy()
    shared["label"] = "normal"
    shared["author_dup"] = 1
    shared["dup"] = rng.integers(2, 9, size=len(shared))
    return pd.concat([rep, shared], ignore_index=True)


def neutral(df):
    """ชุดวัดผลแบบไม่มีข้อมูลซ้ำ (dup=1, author_dup=1) = วัดเฉพาะเนื้อความ"""
    return pd.DataFrame({"text": df["text"], "dup": 1, "author_dup": 1})


def train(csv_path, model_path) -> None:
    df, y = load_labeled(csv_path)
    has_real = {"dup", "author_dup"} <= set(df.columns)
    n_spam, n_ok = int(y.sum()), int((y == 0).sum())
    print(f"ข้อมูลเทรน: spam={n_spam}, ไม่ spam={n_ok}")
    print("ข้อมูลการซ้ำ:", "ใช้ค่าจริงจาก CSV" if has_real else "จำลองขึ้นเอง (ดูหมายเหตุใน train.py)")
    if n_spam < 20 or n_ok < 20:
        print("ตัวอย่างน้อยเกินไป (แนะนำอย่างน้อยประมาณ 50 spam / 50 ไม่ spam) ผลจะไม่น่าเชื่อถือ")
        if min(n_spam, n_ok) < 5:
            sys.exit("ตัวอย่างน้อยกว่า 5 รายการในบางคลาส เทรนไม่ได้")

    rng = np.random.default_rng(RANDOM_STATE)
    df_tr, df_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = fit_spam_model(df_tr, y_tr, rng, has_real)

    # 1) วัดเฉพาะเนื้อความ: ข้อความจริงที่ไม่เคยเห็น ไม่มีข้อมูลซ้ำ
    pred = model.predict(neutral(df_te))
    print("\n=== ผลบนชุดทดสอบ: ดูเฉพาะเนื้อความ (ข้อมูลที่โมเดลไม่เคยเห็น) ===")
    print(classification_report(y_te, pred, target_names=["normal", "spam"], digits=3, zero_division=0))
    cm = confusion_matrix(y_te, pred)
    print("confusion matrix [[TN FP] [FN TP]]:\n", cm)
    report = classification_report(y_te, pred, target_names=["normal", "spam"],
                                   output_dict=True, zero_division=0)

    # 2) เช็คพฤติกรรมการซ้ำ (ข้อความเดียวกัน บัญชีเดียวกัน โพสต์ N ครั้ง)
    probe = df_te[df_te["label"] == "normal"]["text"].tolist()
    probe_short = [t for t in probe if len(t) <= SHORT_LEN] or probe
    print("\n=== ทดสอบพฤติกรรมซ้ำ (ข้อความปกติจากชุดทดสอบ) ===")
    behavior = {}
    for n in (1, 2, 3, 5):
        p = model.predict_proba(pd.DataFrame({"text": probe_short, "dup": n, "author_dup": n}))[:, 1]
        behavior[f"same_author_{n}x_spam_rate"] = round(float((p >= 0.5).mean()), 4)
        print(f"  บัญชีเดียวโพสต์ซ้ำ {n} ครั้ง -> ถูกตัดเป็น spam {100 * (p >= 0.5).mean():.1f}%")
    p = model.predict_proba(pd.DataFrame({"text": probe_short, "dup": 5, "author_dup": 1}))[:, 1]
    behavior["many_authors_5x_spam_rate"] = round(float((p >= 0.5).mean()), 4)
    print(f"  หลายบัญชีพิมพ์ตรงกัน 5 ครั้ง (ไม่มีใครซ้ำเอง) -> spam {100 * (p >= 0.5).mean():.1f}%")

    # 3) cross-validation (เนื้อความอย่างเดียว; เพิ่มข้อมูลจำลองเฉพาะฝั่งเทรนของแต่ละ fold)
    cv_mean = cv_std = None
    cv_k = min(5, n_spam, n_ok)
    if cv_k >= 2:
        scores = []
        skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=RANDOM_STATE)
        for a, b in skf.split(df, y):
            m = fit_spam_model(df.iloc[a], y.iloc[a], rng, has_real)
            scores.append(f1_score(y.iloc[b], m.predict(neutral(df.iloc[b])), zero_division=0))
        cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
        print(f"\nF1 (cross-validation, {cv_k} folds, เนื้อความอย่างเดียว): {cv_mean:.3f} ± {cv_std:.3f}")

    # เทรนใหม่ด้วยข้อมูลทั้งหมดก่อนบันทึก
    final = fit_spam_model(df, y, rng, has_real)
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
        "repeat_features": "real" if has_real else "simulated",
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "test_size": int(len(y_te)),
        "test_spam_text_only": {k: round(float(report["spam"][k]), 4)
                                for k in ("precision", "recall", "f1-score")},
        "test_confusion_matrix_TN_FP_FN_TP": [int(v) for v in cm.ravel()],
        "repeat_behavior": behavior,
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
