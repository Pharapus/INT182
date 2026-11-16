"""เทสต์ส่วน ML: python -m unittest discover -s tests -v   (หรือ  npm run test:py)"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

import pandas as pd  # noqa: E402
from text_utils import MODEL_PATH, clean, dup_counts, dup_features, is_low_quality, load_model, make_frame  # noqa: E402
from predict import predict_texts  # noqa: E402
from train import load_labeled, load_overrides, train  # noqa: E402


class CleanTests(unittest.TestCase):
    def test_strips_html_entities_whitespace(self):
        self.assertEqual(clean("<b>สวัสดี</b>  &quot;ครับ&quot;\n\n"), 'สวัสดี "ครับ"')


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_model_file_exists(self):
        self.assertTrue(Path(MODEL_PATH).exists())

    def test_probabilities_in_range_and_aligned(self):
        texts = ["คลิปดีมากครับ", "😂😂😂", "."]
        probs = predict_texts(texts, self.model)
        self.assertEqual(len(probs), 3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_empty_input(self):
        self.assertEqual(predict_texts([], self.model), [])

    def test_normal_sentence_scores_lower_than_junk(self):
        normal, junk = predict_texts(["ขอบคุณสำหรับคลิปนะครับ ได้ความรู้เยอะมาก", "😂😂😂"], self.model)
        self.assertLess(normal, 0.5)
        self.assertGreaterEqual(junk, 0.5)

    def test_bridge_script_stdin_stdout(self):
        """ml/ml_predict.py คือสิ่งที่ server.js เรียกจริง"""
        items = [{"text": "คลิปดีมากครับ", "dupCount": 1}, {"text": ".", "dupCount": 4}, "สวัสดีครับ"]
        r = subprocess.run([sys.executable, str(ROOT / "ml" / "ml_predict.py")], input=json.dumps(items),
                           capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[1], predict_texts(["."], self.model, [4])[0], places=3)
        self.assertAlmostEqual(out[2], predict_texts(["สวัสดีครับ"], self.model)[0], places=3)

    def test_model_uses_dup_count(self):
        """โมเดลที่ฝึกแล้วต้องรับ dupCount และให้ผลต่างกันเมื่อ dupCount ต่างกัน"""
        a, b = predict_texts(["ข้อความทดสอบ"] * 2, self.model, [1, 2])
        self.assertNotEqual(round(a, 6), round(b, 6))

    def test_metrics_file_matches_model_sklearn(self):
        import sklearn
        m = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(m["sklearn_version"], sklearn.__version__)


class TrainTests(unittest.TestCase):
    def _csv(self, d: Path) -> Path:
        rows = [("สล็อต เว็บตรง ฝากถอน https://a.xyz " + str(i), "spam") for i in range(30)]
        rows += [(f"คลิปนี้ดีมากเลยครับ ตอนที่ {i} สนุกมาก ขอบคุณครับ", "normal") for i in range(60)]
        rows += [("😂", "spam"), ("555", "spam")] * 3
        p = d / "labels.csv"
        pd.DataFrame(rows, columns=["text", "label"]).to_csv(p, index=False, encoding="utf-8-sig")
        return p

    def test_load_labeled_two_labels_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._csv(Path(d))
            df, _, y = load_labeled(p)
            self.assertEqual(set(df["label"]), {"spam", "normal"})
            self.assertEqual(int(y.sum()), int((df["label"] == "spam").sum()))

    def test_load_labeled_skips_legacy_low_quality(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            pd.DataFrame([("a b c d", "normal"), ("สล็อต", "spam"), ("😂", "low_quality")],
                         columns=["text", "label"]).to_csv(p, index=False, encoding="utf-8-sig")
            df, _, _ = load_labeled(p)
            self.assertNotIn("low_quality", set(df["label"]))
            self.assertEqual(len(df), 2)

    def test_is_low_quality_rule(self):
        for t in ["😂", "❤❤❤", "...", "Hi", "สวย", "555", "  "]:
            self.assertTrue(is_low_quality(t), t)
        for t in ["สวยมาก", "Thank you", "Do you have Telegram?"]:
            self.assertFalse(is_low_quality(t), t)

    def test_dup_features_shape_and_clipping(self):
        f = dup_features([[1], [2], [100]])
        self.assertEqual(f.shape, (3, 2))
        self.assertEqual(list(f[:, 0]), [0.0, 1.0, 1.0])
        self.assertEqual(f[0, 1], 0.0)
        self.assertEqual(f[2, 1], f[2, 1])      # ไม่ใช่ NaN
        self.assertLess(f[2, 1], 3.1)            # ถูกจำกัดที่ MAX_DUP

    def test_make_frame_validates_length(self):
        with self.assertRaises(ValueError):
            make_frame(["a", "b"], [1])
        self.assertEqual(list(make_frame(["a"])["dup"]), [1])

    def test_dup_counts_per_video_same_as_web(self):
        rows = [{"text": "Hello!", "videoId": "A"}, {"text": "hello", "videoId": "A"},
                {"text": "hello", "videoId": "B"}, {"text": "1:23", "videoId": "A"}, {"text": "4:56", "videoId": "A"}]
        self.assertEqual(dup_counts(rows), [2, 2, 1, 1, 1])
        self.assertEqual(dup_counts([]), [])

    def test_overrides_replace_labels_and_blank_rows_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            p = self._csv(d)
            ov = d / "ov.csv"
            pd.DataFrame([("😂", "normal", ), ("555", "")], columns=["text", "label"]).to_csv(ov, index=False, encoding="utf-8-sig")
            self.assertEqual(load_overrides(ov), {"😂": "normal"})
            df, _, _ = load_labeled(p, ov)
            self.assertEqual(df.loc[df["text"] == "😂", "label"].iloc[0], "normal")
            self.assertEqual(df.loc[df["text"] == "555", "label"].iloc[0], "spam")   # แถวว่างใช้ป้ายเดิม

    def test_load_labeled_attaches_dup_from_comments(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            p = self._csv(d)
            c = d / "comments.jsonl"
            rows = [{"text": "😂", "videoId": "A"}] * 3 + [{"text": "555", "videoId": "A"}] + [{"text": "ไม่มีในไฟล์นี้", "videoId": "A"}]
            c.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            df, X, _ = load_labeled(p, None, c)
            self.assertEqual(list(X.columns), ["text", "dup"])
            self.assertEqual(int(df.loc[df["text"] == "😂", "dup"].iloc[0]), 3)
            self.assertEqual(int(df.loc[df["text"] == "555", "dup"].iloc[0]), 1)
            df0, _, _ = load_labeled(p)    # ไม่ส่ง comments = ไม่ซ้ำทั้งหมด
            self.assertTrue((df0["dup"] == 1).all())

    def test_train_learns_from_duplicates(self):
        """ถ้าติดป้ายว่าข้อความซ้ำ = spam โมเดลต้องให้คะแนนสูงขึ้นเมื่อซ้ำ"""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows, crow = [], []
            for i in range(40):   # ข้อความหน้าตาปกติ แต่ซ้ำและป้าย spam
                t = f"โปรโมชั่นพิเศษวันนี้เท่านั้น รหัส {i}"
                rows.append((t, "spam")); crow += [{"text": t, "videoId": "A"}] * 3
            for i in range(80):   # ข้อความหน้าตาปกติ ไม่ซ้ำ ป้าย normal
                t = f"โปรโมชั่นพิเศษวันนี้เท่านั้น เรื่องที่ {i + 100} น่าสนใจ"
                rows.append((t, "normal")); crow.append({"text": t, "videoId": "A"})
            csv_p, com = d / "l.csv", d / "c.jsonl"
            pd.DataFrame(rows, columns=["text", "label"]).to_csv(csv_p, index=False, encoding="utf-8-sig")
            com.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in crow), encoding="utf-8")
            out = d / "m.joblib"
            train(csv_p, out, None, com)
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(m["uses_dup_feature"])
            self.assertEqual(m["n_dup_spam_rows"], 40)
            model = load_model(out)
            t = "โปรโมชั่นพิเศษวันนี้เท่านั้น"
            single, dup = predict_texts([t, t], model, [1, 3])
            self.assertGreater(dup, single)

    def test_train_writes_model_and_metrics_to_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._csv(Path(d))
            out = Path(d) / "m.joblib"
            train(p, out)
            self.assertTrue(out.exists())
            metrics = json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["labels"], ["normal", "spam"])
            probs = predict_texts(["สล็อต เว็บตรง ฝากถอน https://z.xyz", "ตอนนี้สนุกมากเลยครับ"], load_model(out))
            self.assertGreater(probs[0], probs[1])


if __name__ == "__main__":
    unittest.main()
