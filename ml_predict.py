# ml_predict.py - รับ JSON array ของข้อความทาง stdin แล้วพิมพ์ JSON array ของความน่าจะเป็นที่เป็น spam
# ใช้โดย server.js  (ต้องมี spam_model.joblib อยู่ข้างๆ ไฟล์นี้ หรือกำหนดพาธด้วย ML_MODEL)
import html
import json
import os
import re
import sys

import joblib

MODEL_FILE = os.environ.get("ML_MODEL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "spam_model.joblib")


def clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(text)))
    return re.sub(r"\s+", " ", text).strip()


model = joblib.load(MODEL_FILE)
texts = [clean(t) for t in json.load(sys.stdin)]
proba = model.predict_proba(texts)[:, 1] if texts else []
print(json.dumps([round(float(p), 4) for p in proba]))
