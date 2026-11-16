"""เทียบผล 2 แบบบนข้อมูลที่ติดป้ายด้วยมือ (ใช้ทำตารางในสไลด์ได้)

  1. ML ใช้ข้อความอย่างเดียว              = ไม่รู้ว่าข้อความซ้ำหรือไม่
  2. ML ใช้ข้อความ + จำนวนครั้งที่ซ้ำ      = โมเดลที่ใช้จริงในเว็บ (ml/train.py)
ทั้งคู่ตัดสินด้วยความน่าจะเป็น >= 0.5 ไม่มีกฎ/threshold ของการซ้ำนอกโมเดล

ใช้ความน่าจะเป็นแบบ out-of-fold (cross_val_predict) คือโมเดลไม่เคยเห็นข้อความนั้นตอนเทรน
จึงเทียบได้ยุติธรรม ไม่ใช่การวัดบนข้อมูลที่ใช้เทรน
นอกจากภาพรวม ยังแยกตัวเลขเฉพาะข้อความที่ซ้ำ (>= 2 ครั้ง) เพราะเป็นส่วนที่ feature นี้ควรช่วย

ต้องมี node ในเครื่อง (ใช้นับข้อความซ้ำ):
    python ml/evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import ROOT  # noqa: E402
from train import COMMENTS_PATH, OVERRIDES_PATH, RANDOM_STATE, build_pipeline, load_labeled  # noqa: E402

ML_THRESHOLD = 0.5


def row(name, y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"method": name,
            "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "to_label.csv"))
    ap.add_argument("--comments", default=str(COMMENTS_PATH))
    ap.add_argument("--overrides", default=str(OVERRIDES_PATH))
    a = ap.parse_args()

    df, X, y = load_labeled(a.data, a.overrides, a.comments)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    dup = (X["dup"] >= 2).to_numpy()

    k = min(5, int(y.sum()), int((y == 0).sum()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    p_text = cross_val_predict(build_pipeline(use_dup=False), X, y, cv=cv, method="predict_proba")[:, 1]
    p_dup = cross_val_predict(build_pipeline(use_dup=True), X, y, cv=cv, method="predict_proba")[:, 1]

    def table(mask, label):
        yy = y[mask]
        if yy.sum() == 0 or mask.sum() == 0:
            return []
        return [row(f"ML ข้อความอย่างเดียว ({label})", yy, p_text[mask] >= ML_THRESHOLD),
                row(f"ML ข้อความ + จำนวนซ้ำ ({label})", yy, p_dup[mask] >= ML_THRESHOLD)]

    results = table(np.ones(len(y), dtype=bool), "ทั้งหมด") + table(dup, "เฉพาะข้อความที่ซ้ำ")
    print(f"ข้อมูล {len(y)} รายการ (positive={int(y.sum())}, ข้อความที่ซ้ำ={int(dup.sum())}, "
          f"spam ที่ซ้ำ={int((dup & (y == 1)).sum())})\n")
    print(f"{'วิธี':<48}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    for r in results:
        print(f"{r['method']:<48}{r['precision']:>7}{r['recall']:>7}{r['f1']:>7}"
              f"{r['TP']:>5}{r['FP']:>5}{r['FN']:>5}")
    out = ROOT / "models" / "evaluation.json"
    out.write_text(json.dumps({"labels": ["normal", "spam"], "n": int(len(y)), "n_positive": int(y.sum()),
                               "n_dup_rows": int(dup.sum()), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกที่ {out}")
