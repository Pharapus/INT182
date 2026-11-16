// รัน: npm test  (node --test) ไม่ต้องใช้ Python / เครือข่าย
import test from "node:test";
import assert from "node:assert/strict";
import { decide, classify, resolveThreshold, MlUnavailableError } from "../app/decide.js";
import { analyze, dupKey } from "../app/duplicates.js";

const item = (text, dupCount = 1) => ({ text, dupCount });

test("resolveThreshold: 0–1 เท่านั้น", () => {
    assert.equal(resolveThreshold(undefined), 0.5);
    assert.equal(resolveThreshold("0.7"), 0.7);
    assert.throws(() => resolveThreshold("1.5"));
    assert.throws(() => resolveThreshold("abc"));
});

test("decide: ML >= threshold เป็น spam", () => {
    assert.equal(decide(0.9), "spam");
    assert.equal(decide(0.5), "spam");      // เท่ากับ threshold นับเป็น spam
    assert.equal(decide(0.49), "normal");
    assert.equal(decide(0.4, { threshold: 0.3 }), "spam");
    assert.equal(decide(0.4, { threshold: 0.8 }), "normal");
});

test("ไม่มีผล ML -> error 503", () => {
    assert.throws(() => decide(null), MlUnavailableError);
});

test("classify: ส่ง text + dupCount ให้โมเดล และตัดสินจาก ML อย่างเดียว (ซ้ำเยอะแต่ ML ต่ำ = normal)", async () => {
    let sent;
    const items = [item("TESTING", 9), item("สล็อต", 1)];
    const out = await classify(items, async x => { sent = x; return [0.11, 0.9]; }, { threshold: 0.5 });
    assert.deepEqual(sent, [{ text: "TESTING", dupCount: 9 }, { text: "สล็อต", dupCount: 1 }]);
    assert.deepEqual(out.map(i => [i.ml, i.label]), [[0.11, "normal"], [0.9, "spam"]]);
    assert.ok(!("spamBy" in out[0]));
});

test("classify: scoreFn ล้ม -> MlUnavailableError status 503", async () => {
    await assert.rejects(
        classify([item("a")], async () => { throw new Error("boom"); }),
        e => e instanceof MlUnavailableError && e.status === 503 && /boom/.test(e.message));
});

test("classify: จำนวนผลไม่ตรงกับจำนวนข้อความ -> error", async () => {
    await assert.rejects(classify([item("a"), item("b")], async () => [0.1]), MlUnavailableError);
});

test("analyze: นับข้อความซ้ำ ไม่สนตัวพิมพ์/เครื่องหมาย/ลิงก์/timestamp และไม่ตัดสินอะไร", () => {
    const rows = [
        { text: "Testing!!" }, { text: "testing" }, { text: "TESTING 1:23 https://x.com" },
        { text: "คลิปดีมาก" },
        { text: "สล็อต เว็บตรง https://abc.xyz แอดไลน์ @abc" }
    ];
    const out = analyze(rows);
    assert.deepEqual(out.map(o => o.dupCount), [3, 3, 3, 1, 1]);
    assert.equal(out[0].dupKey, dupKey("testing"));
    for (const o of out) assert.deepEqual(Object.keys(o).sort(), ["author", "dupCount", "dupKey", "likes", "publishedAt", "text"]);
});

test("analyze: ข้อความที่ตัดแล้วว่าง (timestamp/อีโมจิล้วน) ไม่ถูกเหมารวมเป็นกลุ่มเดียวกัน", () => {
    const out = analyze([{ text: "1:23" }, { text: "4:56" }, { text: "😂" }, { text: "❤" }, { text: "1:23" }, { text: "😂😂😂" }]);
    assert.deepEqual(out.map(o => o.dupCount), [2, 1, 1, 1, 2, 1]);
});
