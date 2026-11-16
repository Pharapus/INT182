"""ฟังก์ชันที่ใช้ร่วมกัน: ทำความสะอาดข้อความ + ตำแหน่งไฟล์โมเดล

สำคัญ: ต้องใช้ clean() ตัวเดียวกันทั้งตอนเทรนและตอนทำนาย
ไม่งั้นข้อความที่โมเดลเห็นจะไม่เหมือนกัน
"""
import html
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_TEXT_LEN = 3  # ข้อความยาว <= ค่านี้ ถือว่าสั้นเกินไป (ใช้เป็นหลักติดป้ายตอนเตรียมข้อมูลเท่านั้น)
MODEL_PATH = Path(os.environ.get("ML_MODEL") or ROOT / "models" / "spam_model.joblib")


def clean(text) -> str:
    """ลบแท็ก HTML -> แปลง entity (&quot; ฯลฯ) -> ยุบช่องว่าง"""
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(text)))
    return re.sub(r"\s+", " ", text).strip()


def load_model(path: Path = MODEL_PATH):
    import joblib

    if not Path(path).exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล {path} ให้รัน  python ml/train.py  ก่อน")
    return joblib.load(path)


def is_symbol_only(text) -> bool:
    """มีแต่เครื่องหมาย/อีโมจิ/ช่องว่าง (ใช้เป็นหลักติดป้ายตอนเตรียมข้อมูล)"""
    return all(unicodedata.category(c)[0] in "PSMC" or c.isspace() for c in str(text))


def is_low_quality(text) -> bool:
    """อีโมจิล้วน หรือสั้น <= MIN_TEXT_LEN ตัวอักษร -> นับเป็น spam; ยาวกว่านั้นและมีตัวอักษร -> normal"""
    t = clean(text)
    return len(t) <= MIN_TEXT_LEN or is_symbol_only(t)


# ---------- จำนวนครั้งที่ข้อความซ้ำ (feature ของโมเดล) ----------
MAX_DUP = 20  # จำนวนซ้ำเกินนี้ถือว่าเท่ากัน กัน feature เดียวครอบงำโมเดล


def dup_features(X):
    """dupCount (คอลัมน์เดียว) -> [ซ้ำหรือไม่ (0/1), log(จำนวนครั้ง)]
    ต้องเป็นฟังก์ชันระดับโมดูลเพื่อให้ joblib บันทึกพร้อมโมเดลได้"""
    d = np.clip(np.asarray(X, dtype=float).reshape(-1), 1, MAX_DUP)
    return np.column_stack([(d >= 2).astype(float), np.log(d)])


def make_frame(texts, dup_counts=None) -> pd.DataFrame:
    """ข้อความ + จำนวนครั้งที่ซ้ำ -> DataFrame (คอลัมน์ text, dup) ที่โมเดลรับ"""
    texts = [clean(t) for t in texts]
    dups = [1] * len(texts) if dup_counts is None else [int(d) for d in dup_counts]
    if len(dups) != len(texts):
        raise ValueError(f"dup_counts {len(dups)} รายการ แต่มีข้อความ {len(texts)} รายการ")
    return pd.DataFrame({"text": texts, "dup": dups})


def dup_counts(rows) -> list:
    """นับข้อความซ้ำ "ต่อวิดีโอ" ด้วย tools/run_dups.mjs (โค้ดชุดเดียวกับเว็บ)
    rows: list ของ dict ที่มี text และ (ถ้ามี) videoId -> list ของจำนวนครั้ง ตรงลำดับกับ rows"""
    if not rows:
        return []
    payload = [{"text": r.get("text", ""), "videoId": r.get("videoId")} for r in rows]
    try:
        r = subprocess.run([os.environ.get("NODE", "node"), str(ROOT / "tools" / "run_dups.mjs"), "-"],
                           input=json.dumps(payload), capture_output=True, text=True,
                           encoding="utf-8", cwd=ROOT)
    except FileNotFoundError:
        raise RuntimeError("ไม่พบ node (ใช้นับข้อความซ้ำ) ติดตั้ง Node.js 20.12+ หรือกำหนดตัวแปร NODE") from None
    if r.returncode != 0:
        raise RuntimeError("นับข้อความซ้ำด้วย node ไม่สำเร็จ: " + r.stderr.strip()[:300])
    return [int(x["dupCount"]) for x in json.loads(r.stdout)]
