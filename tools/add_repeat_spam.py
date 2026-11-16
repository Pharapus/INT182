"""เพิ่มข้อมูลสังเคราะห์ "ข้อความซ้ำ" เพื่อให้โมเดลเรียนรู้เรื่องการซ้ำ (ทาง A)

ทำไมต้องเป็นสคริปต์นี้:
  train.py ตัดข้อความซ้ำใน to_label.csv ทิ้ง (drop_duplicates) และนับจำนวนซ้ำจาก data/comments.jsonl
  "ต่อวิดีโอ" ดังนั้นการเพิ่มแถวซ้ำใน to_label.csv ไม่มีผลเลย ต้องเพิ่มคอมเมนต์ซ้ำลง comments.jsonl
  (videoId เดียวกัน) แล้วติดป้ายข้อความนั้นใน to_label.csv หนึ่งแถวต่อหนึ่งข้อความ

สคริปต์ทำ 2 อย่าง:
  1. ต่อท้าย data/comments.jsonl ด้วยคอมเมนต์ซ้ำ ใช้ videoId เป็น synthetic_NN (แยกจากวิดีโอจริง
     จะได้ไม่ไปเปลี่ยนจำนวนซ้ำของวิดีโอจริง)
  2. ต่อท้าย data/to_label.csv ด้วยข้อความที่ไม่ซ้ำ พร้อมป้าย spam / normal

วิธีใช้:  python tools/add_repeat_spam.py            (รันซ้ำได้ ถ้าเคยเพิ่มแล้วจะไม่เพิ่มซ้ำ)
          python tools/add_repeat_spam.py --remove   (ลบข้อมูลสังเคราะห์ออก)
จากนั้น:  python ml/train.py

แก้/เพิ่ม SPAM_SEEDS ด้วยข้อความ spam จริงที่เจอจะดีที่สุด ข้อมูลสังเคราะห์เป็นแค่ตัวช่วยเริ่มต้น
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from text_utils import ROOT, clean  # noqa: E402

COMMENTS = ROOT / "data" / "comments.jsonl"
LABELS = ROOT / "data" / "to_label.csv"
PREFIX = "synthetic_"

SPAM_SEEDS = [
    "testing", "test test", "testing 123", "ทดสอบๆ", "เทสๆ",
    "Check my profile for free gift cards", "Subscribe to my channel please",
    "Free robux click my channel", "I made $5000 a week working from home, ask me how",
    "Follow me on Instagram @fastgrow_official", "Click the link in my bio for free followers",
    "DM me for crypto signals, guaranteed profit", "Win a free iPhone here bit.ly/xxxx",
    "sub4sub anyone?", "Check out my new cover song on my channel",
    "Join Telegram @profit_signals_vip now", "Earn money online, message me on WhatsApp",
    "First! like if you agree", "Visit my website for cheap followers",
    "แอดไลน์ @abc123 รับงานเสริมวันละ 3000", "ทักไลน์ รับสอนเทรดคริปโต กำไรชัวร์",
    "ช่วยกดติดตามช่องผมหน่อยครับ", "สนใจลงทุนทักแชทได้เลยค่ะ รับรองกำไร",
    "เว็บพนันออนไลน์ ฝากถอนไม่มีขั้นต่ำ", "รับสมัครตัวแทนขายสินค้า รายได้หลักหมื่น ทักเลย",
    "ดูคลิปผมหน่อยครับ ช่องใหม่", "กดติดตามกลับ 100%", "แจกเครดิตฟรี กดลิงก์ในโปรไฟล์",
    "ขายของถูก ทักอินบ็อกซ์", "อยากได้เงินเสริม ทักมาเลย", "ลุ้นรับ iPhone ฟรี กดลิงก์เลย",
    "มาช่วยกดไลค์ช่องผมหน่อยนะครับ", "สอนทำเงินออนไลน์ ฟรี ทักเลย",
    "Looking for investors DM me now", "Best deals on followers and likes, contact me",
    "Make money fast, message me", "ฝากเว็บ แจกโบนัสฟรี", "Please watch my video and subscribe",
    "Hot girls online click here", "Check my channel for amazing content",
]

NORMAL_SEEDS = [
    "ขอบคุณครับ", "Thanks for sharing", "Nice song", "เพลงเพราะมาก", "Love this",
    "Who is watching in 2026", "ชอบมากครับ", "Great video", "ยอดเยี่ยม", "Amazing!",
    "Great explanation, thank you", "ขอบคุณสำหรับวิดีโอดีๆ ครับ", "Still listening in 2026",
    "ดูแล้วมีความสุขมากเลย", "This brings back memories", "Beautiful voice",
]

SPAM_COUNTS = [2, 2, 3, 3, 3, 4, 5, 8, 12]     # spam ซ้ำมีทั้งน้อยและมาก กันโมเดลคิดว่า "ซ้ำเยอะ = ปกติ"
NORMAL_COUNTS = [2, 2, 3, 4, 5, 8]
SUFFIXES = ["", "", "", " 🔥", "!", " 😊", "...", " 🙏"]


def variant(rng, text):
    """ต่างกันแค่ตัวพิมพ์/เครื่องหมาย/อีโมจิ ซึ่ง dupKey ของเว็บจะนับเป็นข้อความเดียวกัน"""
    t = text + rng.choice(SUFFIXES)
    if t.isascii():
        t = rng.choice([t, t.lower(), t.upper(), t.capitalize()])
    return t


def build(rng):
    rows, labels = [], {}
    for kind, seeds, counts in (("spam", SPAM_SEEDS, SPAM_COUNTS), ("normal", NORMAL_SEEDS, NORMAL_COUNTS)):
        for i, seed in enumerate(seeds):
            vid = f"{PREFIX}{kind}_{i // 5:02d}"        # หลายข้อความต่อวิดีโอสังเคราะห์ได้ นับแยกตาม key อยู่แล้ว
            for _ in range(rng.choice(counts)):
                t = variant(rng, seed)
                rows.append({"videoId": vid, "text": t})
                labels[clean(t)] = kind
    rng.shuffle(rows)
    return rows, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true", help="ลบข้อมูลสังเคราะห์ออกจาก comments.jsonl และ to_label.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    lines = [l for l in open(COMMENTS, encoding="utf-8") if l.strip()]
    real = [l for l in lines if not json.loads(l).get("videoId", "").startswith(PREFIX)]
    had = len(real) != len(lines)

    df = pd.read_csv(LABELS, encoding="utf-8-sig")
    df["label"] = df["label"].astype(str)

    if a.remove or had:
        synth_texts = {clean(json.loads(l)["text"]) for l in lines if l not in real}
        df = df[~df["text"].map(clean).isin(synth_texts)]
        with open(COMMENTS, "w", encoding="utf-8") as f:
            f.writelines(real)
        if a.remove:
            df.to_csv(LABELS, index=False, encoding="utf-8-sig")
            print(f"ลบข้อมูลสังเคราะห์แล้ว (comments.jsonl เหลือ {len(real)} บรรทัด)")
            return
        print("พบข้อมูลสังเคราะห์เดิม ลบแล้วสร้างใหม่")

    rows, labels = build(random.Random(a.seed))
    existing = set(df["text"].map(clean))
    clash = [t for t in labels if t in existing]
    if clash:
        # ข้อความตรงกับของจริงที่ติดป้ายไว้แล้ว: ใช้ป้ายเดิม ไม่เขียนทับ
        print(f"ข้ามการติดป้าย {len(clash)} ข้อความที่มีอยู่แล้วใน to_label.csv (ใช้ป้ายเดิม)")
    new = pd.DataFrame([{"text": t, "label": l} for t, l in labels.items() if t not in existing])
    with open(COMMENTS, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.concat([df, new]).to_csv(LABELS, index=False, encoding="utf-8-sig")
    n_sp = sum(1 for v in labels.values() if v == "spam")
    print(f"เพิ่มคอมเมนต์ {len(rows)} บรรทัด ใน comments.jsonl; to_label.csv เพิ่ม {len(new)} ข้อความ "
          f"(spam {n_sp}, normal {len(labels) - n_sp})")


if __name__ == "__main__":
    main()
