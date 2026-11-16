"""สร้างไฟล์ให้ติดป้ายเฉพาะ "ข้อความที่ซ้ำ" จาก data/comments.jsonl

วิธีใช้:  python tools/make_dup_to_label.py
ผลลัพธ์:  data/dup_to_label.csv  คอลัมน์
    text           ข้อความ
    dup_count      จำนวนครั้งที่ซ้ำมากที่สุดในวิดีโอเดียว (นับแบบเดียวกับที่เว็บนับ)
    n_videos       ข้อความนี้พบในกี่วิดีโอ
    current_label  ป้ายเดิมใน to_label.csv (ว่าง = ยังไม่มี)
    label          <- เติมช่องนี้เป็น spam หรือ normal
แล้วรัน python ml/train.py (จะอ่านไฟล์นี้อัตโนมัติ แถวที่ label ว่างจะใช้ป้ายเดิม)

หลักการติดป้าย (ติดที่ "ตัวข้อความ" ไม่ใช่ตัวคอมเมนต์):
  spam    ซ้ำเพราะโพสต์ก็อปวาง/โปรโมต/บอท/ชวนติดต่อ/ปั่น
  normal  ซ้ำเพราะบังเอิญตรงกัน เช่น "ขอบคุณครับ", timestamp "1:23", อีโมจิเดี่ยวๆ ที่คนทั่วไปพิมพ์
ถ้าไม่แน่ใจ ปล่อยว่างไว้ ไม่เป็นไร
ถ้ามีไฟล์อยู่แล้วจะไม่เขียนทับ (เขียนเป็น dup_to_label_new.csv แทน)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from text_utils import ROOT, clean, dup_counts  # noqa: E402

src = ROOT / "data" / "comments.jsonl"
labels_file = ROOT / "data" / "to_label.csv"
out = ROOT / "data" / "dup_to_label.csv"

rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
df = pd.DataFrame({
    "text": [clean(r.get("text", "")) for r in rows],
    "video": [r.get("videoId") for r in rows],
    "dup_count": dup_counts(rows),
})
df = df[df["text"] != ""]
g = df.groupby("text").agg(dup_count=("dup_count", "max"), n_videos=("video", "nunique")).reset_index()
g = g[g["dup_count"] >= 2]

current = {}
if labels_file.exists():
    lab = pd.read_csv(labels_file, encoding="utf-8-sig")
    lab["label"] = lab["label"].astype(str).str.strip().str.lower()
    current = {clean(t): l for t, l in zip(lab["text"], lab["label"]) if l in {"spam", "normal"}}
g["current_label"] = g["text"].map(current).fillna("")
g["label"] = ""
g = g.sort_values(["dup_count", "text"], ascending=[False, True])[["text", "dup_count", "n_videos", "current_label", "label"]]

if out.exists():
    out = ROOT / "data" / "dup_to_label_new.csv"
    print(f"มี dup_to_label.csv อยู่แล้ว จึงเขียนเป็น {out.name} แทน")
g.to_csv(out, index=False, encoding="utf-8-sig")
print(len(g), "ข้อความที่ซ้ำ ->", out)
print("ป้ายเดิม:", g["current_label"].replace("", "(ยังไม่มี)").value_counts().to_dict())
