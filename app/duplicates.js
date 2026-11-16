// นับข้อความซ้ำ (ไม่มีกฎให้คะแนน/ตัดสินใดๆ) ผลคือ dupCount ที่ส่งเข้าโมเดล ML เป็น feature
// ตัวโมเดลเป็นผู้เรียนรู้เองว่าซ้ำแบบไหนคือ spam ดู ml/train.py

const decode = s => s
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

const clean = raw => decode(String(raw).replace(/<[^>]*>/g, " ")).replace(/\s+/g, " ").trim();

// คีย์เทียบข้อความซ้ำ: ตัด timestamp/ลิงก์/เครื่องหมาย/อีโมจิ/ช่องว่าง, ตัวพิมพ์เล็ก, ยุบตัวอักษรซ้ำติดกันเหลือ 2
// ถ้าตัดแล้วไม่เหลืออะไร (เช่น "1:23" หรืออีโมจิล้วน) ใช้ข้อความเดิมที่ตัดช่องว่างแทน
// จะได้ไม่เหมารวมคอมเมนต์ที่ต่างกันเป็นกลุ่มซ้ำเดียวกัน
export function dupKey(t) {
    const key = t.toLowerCase()
        .replace(/https?:\/\/\S+/g, "")
        .replace(/\b\d{1,2}:\d{2}(:\d{2})?\b/g, "")
        .replace(/[^\p{L}\p{N}]/gu, "")
        .replace(/(.)\1{2,}/gu, "$1$1");
    return key || t.toLowerCase().replace(/\s+/g, "");
}

/** comments: [{text, authorDisplayName, likeCount, publishedAt}] -> รายการพร้อม dupCount / dupKey */
export function analyze(comments) {
    const prep = comments.map(c => {
        const t = clean(c.text ?? "");
        return { ...c, t, key: dupKey(t) };
    });

    const counts = new Map();   // key -> จำนวนทั้งหมดในชุดที่ดึงมา
    for (const p of prep) counts.set(p.key, (counts.get(p.key) || 0) + 1);

    return prep.map(p => ({
        text: p.t,
        author: p.authorDisplayName ?? "",
        likes: p.likeCount ?? 0,
        publishedAt: p.publishedAt ?? null,
        dupCount: counts.get(p.key) || 1,   // นับรวมตัวเอง
        dupKey: p.key                       // key เดียวกัน = กลุ่มซ้ำเดียวกัน
    }));
}
