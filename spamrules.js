// กฎแยก spam (ชุดเดียวกับ classify.js + กฎ "บัญชีเดียวโพสต์ข้อความเดิมซ้ำ")
export const SPAM_THRESHOLD = 3;

const decode = s => s
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

const clean = raw => decode(String(raw).replace(/<[^>]*>/g, " ")).replace(/\s+/g, " ").trim();

function links(raw) {
    const d = decode(String(raw));
    const withScheme = d.match(/https?:\/\/[^\s"'<>]+/gi) || [];
    const bare = d.replace(/<[^>]*>/g, " ")
        .match(/\b[a-z0-9-]{3,}\.(com|net|org|xyz|top|vip|bet|cc|me|info|co|io|site|online|click)\b/gi) || [];
    return [...withScheme, ...bare];
}

function isExternal(link) {
    try {
        const host = new URL(link.startsWith("http") ? link : "https://" + link)
            .hostname.replace(/^www\./, "");
        return !["youtube.com", "m.youtube.com", "youtu.be"].includes(host);
    } catch {
        return true;
    }
}

const GAMBLING = /(สล็อต|บาคาร่า|แทงบอล|หวยออนไลน์|เว็บตรง|เว็บพนัน|คาสิโน|ฝากถอน|เครดิตฟรี|ufabet|pg\s?slot|slot)/i;
const MONEY = /(รายได้เสริม|ทำเงินออนไลน์|งานออนไลน์|รับงาน|ปล่อยกู้|กู้เงิน|ปั่นยอด|รับปั่น|ทำเงินวันละ|forex|ฟอเร็กซ์)/i;
const PROMO = /(คุณได้รับรางวัล|ยินดีด้วย.{0,20}(รางวัล|ผู้โชคดี)|ผู้โชคดี|giveaway|subscribe|ซับกลับ|sub\s?back|ดูช่องผม|เข้าช่องผม|check (out )?my channel|ช่วยกดติดตาม)/i;
const CONTACT = /(t\.me\/|telegram|เทเลแกรม|whatsapp|วอทแอป|แอด\s?ไลน์|add\s?line|line\s*(id|:|@)|ไลน์\s*(id|:|@)|ทักแชท|ทักมา|ทักหา|inbox|อินบ็อกซ์)/i;

const RULES = [
    ["ลิงก์ภายนอก", 3, c => c.ext.length > 0],
    ["ชวนติดต่อ", 2, c => CONTACT.test(c.t)],
    ["@handle", 2, c => /(^|\s)@[\w.]{3,}/.test(c.t)],
    ["เบอร์โทร", 2, c => /\b0\d{1,2}[- ]?\d{3}[- ]?\d{4}\b/.test(c.t)],
    ["พนัน", 3, c => GAMBLING.test(c.t)],
    ["ชวนทำเงิน", 2, c => MONEY.test(c.t)],
    ["โปรโมต/แจกรางวัล", 2, c => PROMO.test(c.t)],
    ["ข้อความซ้ำ", 2, c => c.n.length >= 10 && c.dup >= 3],
    ["บัญชีเดียวโพสต์ซ้ำ", 2, c => c.n.length >= 10 && c.authorDup >= 2]
];

const LOW_QUALITY = /^[\p{P}\p{S}\p{M}\p{C}\s]*$/u; // มีแต่เครื่องหมาย/อีโมจิ

/** comments: [{text, authorChannelId, authorDisplayName, likeCount, publishedAt}] */
export function analyze(comments) {
    const prep = comments.map(c => {
        const t = clean(c.text ?? "");
        return { ...c, raw: c.text ?? "", t, n: t.toLowerCase() };
    });

    const byText = new Map();
    const byAuthor = new Map();
    for (const p of prep) {
        byText.set(p.n, (byText.get(p.n) || 0) + 1);
        if (p.authorChannelId) {
            const k = p.authorChannelId + "|" + p.n;
            byAuthor.set(k, (byAuthor.get(k) || 0) + 1);
        }
    }

    return prep.map(p => {
        const ctx = {
            t: p.t,
            n: p.n,
            ext: links(p.raw).filter(isExternal),
            dup: byText.get(p.n) || 1,
            authorDup: p.authorChannelId ? byAuthor.get(p.authorChannelId + "|" + p.n) : 1
        };
        let score = 0;
        const reasons = [];
        for (const [name, weight, test] of RULES) {
            if (test(ctx)) { score += weight; reasons.push(name); }
        }
        return {
            text: p.t,
            author: p.authorDisplayName ?? "",
            likes: p.likeCount ?? 0,
            publishedAt: p.publishedAt ?? null,
            score,
            reasons,
            dup: ctx.dup,
            authorDup: ctx.authorDup,
            lowQuality: p.t.length <= 3 || LOW_QUALITY.test(p.t)
        };
    });
}

export function labelOf(item, threshold = SPAM_THRESHOLD) {
  if (item.score >= threshold) return "spam";
  if (item.lowQuality || item.score >= 1) return "low_quality";
  return "normal";
}
