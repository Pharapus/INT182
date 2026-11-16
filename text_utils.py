"""ฟังก์ชันที่ใช้ร่วมกันระหว่างเทรน (train.py) และทำนาย (ml_predict.py)

สำคัญ: ต้องใช้ clean() และ repeat_transform() ตัวเดียวกันทั้งตอนเทรนและตอนทำนาย
ไม่งั้นข้อมูลที่โมเดลเห็นจะไม่เหมือนกัน
"""
import html
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("ML_MODEL") or ROOT / "spam_model.joblib")

# คอลัมน์ที่โมเดลรับเข้า: text, dup, author_dup
#   dup        = ข้อความนี้ (ตัวพิมพ์เล็ก/ยุบช่องว่างแล้ว) โผล่กี่ครั้งในวิดีโอเดียวกัน (รวมทุกบัญชี)
#   author_dup = บัญชีเดียวกันโพสต์ข้อความนี้กี่ครั้งในวิดีโอเดียวกัน
FEATURE_COLUMNS = ["text", "dup", "author_dup"]
REPEAT_COLUMNS = ["dup", "author_dup"]
REPEAT_SCALE = 3.0   # ขยายขนาด feature ให้สู้กับ TF-IDF (ซึ่งถูก normalize แล้ว) ได้


def clean(text) -> str:
    """ลบแท็ก HTML -> แปลง entity (&quot; ฯลฯ) -> ยุบช่องว่าง"""
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(text)))
    return re.sub(r"\s+", " ", text).strip()


def repeat_transform(X):
    """นับครั้งที่ซ้ำ (ไม่รวมครั้งแรก) -> log -> ขยาย  (1 ครั้ง=0, 2 ครั้ง=0.69*scale, ...)"""
    arr = np.asarray(X, dtype=float)
    return REPEAT_SCALE * np.log1p(np.clip(arr - 1, 0, 9))


def meta_features(tlogit, dup, author_dup):
    """ฟีเจอร์ของโมเดลชั้นที่ 2: [คะแนนเนื้อความ (logit), ซ้ำข้ามบัญชี, บัญชีเดียวซ้ำ]"""
    rep = repeat_transform(np.column_stack([dup, author_dup]))
    return np.column_stack([np.asarray(tlogit, dtype=float), rep])


class SpamModel:
    """โมเดล 2 ชั้น

    ชั้น 1 text_model  : TF-IDF char n-gram + Logistic Regression ดูเฉพาะเนื้อความ
    ชั้น 2 meta_model  : Logistic Regression เล็กๆ รวมคะแนนเนื้อความ + พฤติกรรมโพสต์ซ้ำ

    แยกชั้นกันเพื่อให้ข้อมูลพฤติกรรม (ที่จำลอง) ไม่ไปรบกวนการเรียนรู้เนื้อความจริง
    """

    def __init__(self, text_model, meta_model):
        self.text_model = text_model
        self.meta_model = meta_model

    def text_logit(self, texts):
        p = np.clip(self.text_model.predict_proba(list(texts))[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict_proba(self, df):
        X = meta_features(self.text_logit(df["text"]), df["dup"], df["author_dup"])
        return self.meta_model.predict_proba(X)

    def predict(self, df):
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)


def to_frame(items) -> pd.DataFrame:
    """แปลงรายการที่ server ส่งมา -> DataFrame ที่โมเดลใช้

    แต่ละรายการเป็น string (ไม่มีข้อมูลซ้ำ = 1) หรือ dict {text, dup, authorDup}
    """
    rows = []
    for it in items:
        if isinstance(it, dict):
            rows.append((clean(it.get("text", "")),
                         max(1, int(it.get("dup") or 1)),
                         max(1, int(it.get("authorDup") or 1))))
        else:
            rows.append((clean(it), 1, 1))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
