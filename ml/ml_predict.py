"""สะพานให้ app/server.js เรียกใช้โมเดล

รับ JSON array ทาง stdin แล้วพิมพ์ JSON array ของความน่าจะเป็นที่เป็น spam
แต่ละรายการเป็น {"text": "...", "dupCount": จำนวนครั้งที่ซ้ำ} (หรือเป็นข้อความล้วน = ไม่ซ้ำ)
ตัวอย่าง:  echo '[{"text":"คลิปดีมาก","dupCount":1},"สนใจงานแอดไลน์"]' | python ml/ml_predict.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import load_model, make_frame  # noqa: E402

model = load_model()
items = json.load(sys.stdin)
texts = [i if isinstance(i, str) else i.get("text", "") for i in items]
dups = [1 if isinstance(i, str) else i.get("dupCount", 1) for i in items]
proba = model.predict_proba(make_frame(texts, dups))[:, 1] if items else []
print(json.dumps([round(float(p), 4) for p in proba]))
