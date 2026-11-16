// ตัดสินว่าคอมเมนต์เป็น spam หรือไม่ แยกออกมาจาก server.js เพื่อให้เทสต์ได้โดยไม่ต้องเปิดเซิร์ฟเวอร์
//
// ML ล้วนๆ: spam เมื่อความน่าจะเป็นจากโมเดล >= ML_THRESHOLD
// โมเดลรับทั้งข้อความและ dupCount (จำนวนครั้งที่ข้อความนี้ซ้ำในชุดที่ดึงมา) เป็นอินพุต
// ไม่มีกฎ/threshold ของการซ้ำนอกโมเดล ถ้าเรียกโมเดลไม่ได้จะ "ล้มเหลวชัดเจน" (HTTP 503)

export const DEFAULT_ML_THRESHOLD = 0.5;

export class MlUnavailableError extends Error {
    constructor(cause) {
        super("โมเดล ML ใช้ไม่ได้ ตรวจว่าติดตั้ง Python + pip install -r requirements.txt และมี models/spam_model.joblib แล้ว"
            + (cause ? ` [${cause}]` : ""));
        this.name = "MlUnavailableError";
        this.status = 503;
    }
}

export function resolveThreshold(raw) {
    if (raw === undefined || raw === null || String(raw).trim() === "") return DEFAULT_ML_THRESHOLD;
    const t = Number(raw);
    if (!Number.isFinite(t) || t < 0 || t > 1) throw new Error(`ML_THRESHOLD="${raw}" ต้องเป็นตัวเลข 0–1`);
    return t;
}

export function decide(ml, { threshold = DEFAULT_ML_THRESHOLD } = {}) {
    if (ml == null) throw new MlUnavailableError();
    return ml >= threshold ? "spam" : "normal";
}

// items = ผลจาก analyze() (มี text, dupCount)
// scoreFn([{text, dupCount}]) -> Promise<number[]> (ในเซิร์ฟเวอร์คือ mlScores ที่เรียก ml/ml_predict.py)
export async function classify(items, scoreFn, opts = {}) {
    let probs;
    try {
        probs = await scoreFn(items.map(i => ({ text: i.text, dupCount: i.dupCount ?? 1 })));
        if (!Array.isArray(probs) || probs.length !== items.length) {
            throw new Error(`โมเดลคืนผล ${Array.isArray(probs) ? probs.length : "ผิดรูปแบบ"} รายการ แต่ส่งไป ${items.length}`);
        }
    } catch (e) {
        throw new MlUnavailableError(e.message);
    }
    return items.map((it, k) => ({ ...it, ml: probs[k], label: decide(probs[k], opts) }));
}
