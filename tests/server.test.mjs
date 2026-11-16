// เทสต์ปลายทางถึงปลายทาง: เปิด YouTube API จำลอง + เปิด server.js จริง (เรียกโมเดล Python จริง)
// ต้องมี python3 + pip install -r requirements.txt และ models/spam_model.joblib
import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const VIDEO = "abcdefghijk"; // 11 ตัวอักษร

const COMMENTS = [
    "คลิปดีมากครับ ชอบมากเลย รอตอนต่อไป",
    "ขอบคุณสำหรับคลิปนะครับ ได้ความรู้เยอะมาก",
    "คลิปนี้ดีมากเลยครับ ซ้ำกันสองรอบ",
    "คลิปนี้ดีมากเลยครับ ซ้ำกันสองรอบ",
    "😂😂😂",
    ".",
    "เว็บตรง สล็อต ฝากถอน เครดิตฟรี สมัครเลย https://abc.xyz https://def.top"
];

const listen = srv => new Promise(r => srv.listen(0, "127.0.0.1", () => r(srv.address().port)));

function fakeYouTube() {
    return http.createServer((req, res) => {
        const u = new URL(req.url, "http://x");
        res.setHeader("Content-Type", "application/json");
        if (u.pathname.endsWith("/videos")) {
            return res.end(JSON.stringify({ items: [{ snippet: { title: "วิดีโอทดสอบ", channelTitle: "ช่องทดสอบ" }, statistics: { commentCount: String(COMMENTS.length) } }] }));
        }
        if (u.pathname.endsWith("/commentThreads")) {
            return res.end(JSON.stringify({
                items: COMMENTS.map((t, i) => ({ snippet: { topLevelComment: { snippet: {
                    textOriginal: t, authorChannelId: { value: "UC" + i }, authorDisplayName: "user" + i, likeCount: i, publishedAt: "2026-11-15T00:00:00Z"
                } } } }))
            }));
        }
        res.statusCode = 404; res.end("{}");
    });
}

function startApp(env) {
    return new Promise((resolve, reject) => {
        const p = spawn(process.execPath, [path.join(root, "app", "server.js")], {
            cwd: root, env: { ...process.env, YT_API_KEY: "test-key", ...env }
        });
        let out = "";
        const timer = setTimeout(() => reject(new Error("server start timeout: " + out)), 8000);
        p.stdout.on("data", d => {
            out += d;
            if (/เปิดที่ http/.test(out)) { clearTimeout(timer); resolve(p); }
        });
        p.on("exit", code => { clearTimeout(timer); reject(new Error(`server exited ${code}: ${out}`)); });
    });
}

const post = (port, body) => fetch(`http://127.0.0.1:${port}/api/analyze`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
});

let yt, ytPort, appMl, appBroken, appMlPort = 3101, appBrokenPort = 3103;

before(async () => {
    yt = fakeYouTube();
    ytPort = await listen(yt);
    const base = { YT_API_URL: `http://127.0.0.1:${ytPort}` };
    appMl = await startApp({ ...base, PORT: appMlPort });
    // PYTHON ชี้ไปที่คำสั่งที่ไม่มีอยู่ เพื่อจำลองว่าเรียกโมเดลไม่ได้
    appBroken = await startApp({ ...base, PORT: appBrokenPort, PYTHON: "python-does-not-exist" });
});

after(() => {
    for (const p of [appMl, appBroken]) p?.removeAllListeners("exit");
    for (const p of [appMl, appBroken]) p?.kill();
    yt?.close();
});

test("ตัดสินด้วย ML ล้วนๆ คืน ml/label/dupCount ครบ ไม่มีคะแนนกฎ/dupThreshold", async () => {
    const r = await post(appMlPort, { url: `https://www.youtube.com/watch?v=${VIDEO}` });
    assert.equal(r.status, 200);
    const d = await r.json();
    assert.equal(d.mlUsed, true);
    assert.ok(!("dupThreshold" in d) && !("mode" in d));
    assert.equal(d.items.length, COMMENTS.length);
    for (const it of d.items) {
        assert.equal(typeof it.ml, "number");
        assert.ok(it.ml >= 0 && it.ml <= 1);
        assert.ok(!("score" in it) && !("reasons" in it) && !("spamBy" in it));
        assert.equal(it.label, it.ml >= d.mlThreshold ? "spam" : "normal"); // label ตาม ML ล้วนๆ ไม่มีการบังคับเพราะซ้ำ
    }
    const byText = Object.fromEntries(d.items.map(i => [i.text, i]));
    assert.equal(byText["คลิปดีมากครับ ชอบมากเลย รอตอนต่อไป"].label, "normal");
    assert.equal(byText["."].label, "spam");
});

test("ข้อความซ้ำถูกส่งเข้าโมเดลเป็น feature (dupCount = 2 และ ML ต่างจากตอนไม่ซ้ำ)", async () => {
    const d = await (await post(appMlPort, { url: `https://youtu.be/${VIDEO}` })).json();
    const dups = d.items.filter(i => i.text === "คลิปนี้ดีมากเลยครับ ซ้ำกันสองรอบ");
    assert.equal(dups.length, 2);
    assert.ok(dups.every(x => x.dupCount === 2));
    // ถามโมเดลตรงๆ ด้วยข้อความเดียวกันแต่บอกว่าไม่ซ้ำ ต้องได้ค่าต่างกัน แปลว่าโมเดลใช้ dupCount จริง
    const { spawnSync } = await import("node:child_process");
    const r = spawnSync(process.env.PYTHON || "python3", [path.join(root, "ml", "ml_predict.py")], {
        cwd: root, encoding: "utf-8",
        input: JSON.stringify([{ text: dups[0].text, dupCount: 1 }, { text: dups[0].text, dupCount: 2 }])
    });
    assert.equal(r.status, 0, r.stderr);
    const [single, twice] = JSON.parse(r.stdout);
    assert.notEqual(single, twice);
    assert.equal(twice, dups[0].ml);
});

test("เรียกโมเดลไม่ได้ -> 503 พร้อมข้อความชัดเจน", async () => {
    const r = await post(appBrokenPort, { url: `https://youtu.be/${VIDEO}` });
    assert.equal(r.status, 503);
    assert.match((await r.json()).error, /โมเดล ML ใช้ไม่ได้/);
});

test("ลิงก์ไม่ใช่ YouTube -> 400", async () => {
    const r = await post(appMlPort, { url: "https://example.com/x" });
    assert.equal(r.status, 400);
});

test("ML_THRESHOLD ไม่ถูกต้อง -> เซิร์ฟเวอร์ปฏิเสธการเริ่มทำงาน", async () => {
    await assert.rejects(startApp({ PORT: 3104, ML_THRESHOLD: "2" }), /ตั้งค่าไม่ถูกต้อง|exited 1/);
});
