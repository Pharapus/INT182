// นับข้อความซ้ำ (app/duplicates.js) "ต่อวิดีโอ" แล้วพิมพ์ผลเป็น JSON ทาง stdout (ลำดับตรงกับแถวนำเข้า)
//   node tools/run_dups.mjs [data/comments.jsonl]     อ่านจากไฟล์ .jsonl
//   echo '[{"text":"..","videoId":".."}]' | node tools/run_dups.mjs -    อ่าน JSON array จาก stdin
// ใช้โดย ml/text_utils.py (dup_counts) ตอนเทรน/วัดผล/ทำนายไฟล์ ให้นับด้วยโค้ดชุดเดียวกับเว็บ
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { analyze } from "../app/duplicates.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = process.argv[2] || path.join(root, "data", "comments.jsonl");
const raw = arg === "-" ? fs.readFileSync(0, "utf8") : fs.readFileSync(arg, "utf8");
const rows = arg === "-" ? JSON.parse(raw) : raw.split("\n").filter(l => l.trim()).map(l => JSON.parse(l));

const groups = new Map();   // videoId -> [index แถว] (เว็บนับทีละวิดีโอ จึงนับแบบเดียวกัน)
rows.forEach((r, i) => {
    const k = r.videoId ?? "";
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(i);
});
const out = new Array(rows.length);
for (const idx of groups.values()) {
    analyze(idx.map(i => ({ text: rows[i].text }))).forEach((a, j) => { out[idx[j]] = { text: a.text, dupCount: a.dupCount }; });
}
process.stdout.write(JSON.stringify(out));
