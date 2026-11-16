"""ฟังก์ชันที่ใช้ร่วมกันระหว่างเทรน (train.py) และทำนาย (ml_predict.py)

สำคัญ: ต้องใช้ clean() ตัวเดียวกันทั้งตอนเทรนและตอนทำนาย
ไม่งั้นข้อความที่โมเดลเห็นจะไม่เหมือนกัน
"""
import html
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("ML_MODEL") or ROOT / "spam_model.joblib")


def clean(text) -> str:
    """ลบแท็ก HTML -> แปลง entity (&quot; ฯลฯ) -> ยุบช่องว่าง"""
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(text)))
    return re.sub(r"\s+", " ", text).strip()
