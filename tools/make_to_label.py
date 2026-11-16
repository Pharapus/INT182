"""สร้างไฟล์ให้ติดป้ายด้วยมือ จาก data/comments.jsonl

วิธีใช้:  python tools/make_to_label.py
ผลลัพธ์:  data/to_label.csv  (คอลัมน์ text, label)  ถ้ามีอยู่แล้วจะเขียนเป็น data/to_label_new.csv
          แล้วเปิดด้วย Excel/Google Sheets เติมคอลัมน์ label เป็น  spam / normal
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from text_utils import ROOT, clean  # noqa: E402

src = ROOT / "data" / "comments.jsonl"
out = ROOT / "data" / "to_label.csv"

rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
df = pd.DataFrame({"text": [clean(r.get("text", "")) for r in rows]})
df = df[df["text"] != ""].drop_duplicates(subset="text")
df["label"] = ""

if out.exists():  # กันเขียนทับไฟล์ที่ติดป้ายไว้แล้ว
    out = ROOT / "data" / "to_label_new.csv"
    print(f"มี to_label.csv อยู่แล้ว จึงเขียนเป็น {out.name} แทน")

df.to_csv(out, index=False, encoding="utf-8-sig")
print(len(df), "แถว ->", out)
