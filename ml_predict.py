# ml_predict.py - รับ JSON array ทาง stdin แล้วพิมพ์ JSON array ของความน่าจะเป็นที่เป็น spam
# แต่ละรายการเป็นข้อความ (string) หรือ object {"text": "...", "dup": 1, "authorDup": 1}
#   dup        = ข้อความเดียวกันซ้ำกี่ครั้งในวิดีโอ   authorDup = บัญชีเดียวกันโพสต์ซ้ำกี่ครั้ง
# ใช้โดย server.js  (ต้องมี spam_model.joblib อยู่ข้างๆ ไฟล์นี้ หรือกำหนดพาธด้วย ML_MODEL)
# ถ้ายังไม่มีโมเดลหรืออยากเทรนใหม่ ให้รัน  python train.py
import json
import sys

import joblib

from text_utils import MODEL_PATH, to_frame

model = joblib.load(MODEL_PATH)
items = json.load(sys.stdin)
proba = model.predict_proba(to_frame(items))[:, 1] if items else []
print(json.dumps([round(float(p), 4) for p in proba]))
