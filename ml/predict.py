"""ทำนายคอมเมนต์จากไฟล์ แล้วเขียนผลออกมาเป็น jsonl

วิธีใช้:
    python ml/predict.py data/comments.jsonl                 -> data/predicted.jsonl
    python ml/predict.py comments.csv -o out.jsonl -t 0.6

ไฟล์นำเข้า: .jsonl (แต่ละบรรทัดมีฟิลด์ text และถ้ามี videoId) หรือ .csv (มีคอลัมน์ text)
โมเดลใช้ทั้งข้อความและจำนวนครั้งที่ซ้ำ (นับต่อ videoId ด้วย node) ต้องมี Node.js
ผลลัพธ์เพิ่มฟิลด์: dup_count, spam_prob (ความน่าจะเป็น 0-1) และ predicted (spam / normal)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import MODEL_PATH, ROOT, clean, dup_counts, load_model, make_frame  # noqa: E402


def read_rows(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig").to_dict("records")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def predict_texts(texts, model=None, dup_counts=None):
    """คืนรายการความน่าจะเป็น spam ของแต่ละข้อความ (ใช้เรียกจากโค้ดอื่นได้)
    dup_counts = จำนวนครั้งที่แต่ละข้อความซ้ำ (ไม่ใส่ = 1 คือไม่ซ้ำ)"""
    model = model or load_model()
    texts = list(texts)
    return [float(p) for p in model.predict_proba(make_frame(texts, dup_counts))[:, 1]] if texts else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=str(ROOT / "data" / "predicted.jsonl"))
    ap.add_argument("-t", "--threshold", type=float, default=0.5)
    ap.add_argument("--model", default=str(MODEL_PATH))
    a = ap.parse_args()

    rows = read_rows(Path(a.input))
    model = load_model(Path(a.model))
    dups = dup_counts(rows)
    probs = predict_texts([r.get("text", "") for r in rows], model, dups)
    with open(a.output, "w", encoding="utf-8") as f:
        for r, p, d in zip(rows, probs, dups):
            r.update(text=clean(r.get("text", "")), dup_count=d, spam_prob=round(p, 4),
                     predicted="spam" if p >= a.threshold else "normal")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_spam = sum(p >= a.threshold for p in probs)
    print(f"ทำนาย {len(rows)} คอมเมนต์ -> {a.output}  (spam {n_spam})")
