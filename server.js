// รัน:  npm start   แล้วเปิด http://localhost:3000   (คีย์อยู่ในไฟล์ .env: YT_API_KEY=...)
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import axios from "axios";
import { analyze, SPAM_THRESHOLD } from "./spamrules.js";

const dir = path.dirname(fileURLToPath(import.meta.url));

// อ่านค่าจากไฟล์ .env ข้างๆ server.js (ถ้ามี) ค่าที่ตั้งใน environment จะมาก่อน
try { process.loadEnvFile(path.join(dir, ".env")); } catch { /* ไม่มีไฟล์ .env ก็ข้าม */ }

const PORT = Number(process.env.PORT || 3000);
const API_KEY = process.env.YT_API_KEY;
const YT_API = process.env.YT_API_URL || "https://www.googleapis.com/youtube/v3";
const DEFAULT_MAX = 500;
const HARD_MAX = 2000; // กันกิน quota เกินไป (100 comments = 1 quota unit)

// หา index.html ใน public/ ก่อน ถ้าไม่มีลองหาข้างๆ server.js
const INDEX_PATHS = [path.join(dir, "public", "index.html"), path.join(dir, "index.html")];
const loadIndex = () => {
    const p = INDEX_PATHS.find(fs.existsSync);
    return p ? fs.readFileSync(p) : null;
};

if (!API_KEY) {
    console.log("ยังไม่ได้ตั้งค่า YT_API_KEY ให้สร้างไฟล์ .env ข้างๆ server.js โดยมีบรรทัดเดียวว่า");
    console.log("  YT_API_KEY=คีย์ของคุณ");
    process.exit(1);
}

function getVideoId(raw) {
    let u;
    try { u = new URL(String(raw).trim()); } catch { return null; }
    const host = u.hostname.replace(/^www\.|^m\./, "");
    let id = null;
    if (host === "youtu.be") id = u.pathname.slice(1).split("/")[0];
    else if (host === "youtube.com") {
        if (u.pathname === "/watch") id = u.searchParams.get("v");
        else id = (u.pathname.match(/^\/(shorts|live|embed)\/([\w-]{11})/) || [])[2];
    }
    return id && /^[\w-]{11}$/.test(id) ? id : null;
}

const ERRORS = {
    commentsDisabled: [403, "วิดีโอนี้ปิดการแสดงความคิดเห็น"],
    videoNotFound: [404, "ไม่พบวิดีโอนี้ (อาจเป็นวิดีโอส่วนตัวหรือถูกลบ)"],
    quotaExceeded: [429, "โควตา YouTube API วันนี้หมดแล้ว ลองใหม่พรุ่งนี้"],
    keyInvalid: [500, "API key ไม่ถูกต้อง ตรวจค่า YT_API_KEY"],
    forbidden: [403, "ไม่มีสิทธิ์ดึงคอมเมนต์ของวิดีโอนี้"]
};

function mapError(err) {
    const reason = err.response?.data?.error?.errors?.[0]?.reason;
    const message = err.response?.data?.error?.message || "";
    if (reason && ERRORS[reason]) return ERRORS[reason];
    if (/API key not valid/i.test(message)) return ERRORS.keyInvalid;
    if (err.response) return [502, `YouTube API ตอบกลับผิดพลาด (${err.response.status})`];
    return [500, "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"];
}

async function fetchVideoInfo(videoId) {
    const { data } = await axios.get(`${YT_API}/videos`, {
        params: { part: "snippet,statistics", id: videoId, key: API_KEY }
    });
    const v = data.items?.[0];
    if (!v) { const e = new Error("not found"); e.response = { data: { error: { errors: [{ reason: "videoNotFound" }] } } }; throw e; }
    return {
        id: videoId,
        title: v.snippet.title,
        channel: v.snippet.channelTitle,
        commentCount: Number(v.statistics?.commentCount ?? 0)
    };
}

async function fetchComments(videoId, max) {
    const out = [];
    let pageToken;
    do {
        const { data } = await axios.get(`${YT_API}/commentThreads`, {
            params: {
                part: "snippet", videoId, maxResults: 100, pageToken,
                order: "time", textFormat: "plainText", key: API_KEY
            }
        });
        for (const item of data.items) {
            const s = item.snippet.topLevelComment.snippet;
            out.push({
                text: s.textOriginal ?? s.textDisplay,
                authorChannelId: s.authorChannelId?.value ?? null,
                authorDisplayName: s.authorDisplayName,
                likeCount: s.likeCount,
                publishedAt: s.publishedAt
            });
        }
        pageToken = data.nextPageToken;
    } while (pageToken && out.length < max);
    return out.slice(0, max);
}

// ---- โมเดล ML (ml_predict.py + spam_model.joblib) ถ้าใช้ไม่ได้จะถอยกลับไปใช้กฎอย่างเดียว ----
const PYTHON = process.env.PYTHON || "python3";
const ML_THRESHOLD = Number(process.env.ML_THRESHOLD || 0.5);

function mlScores(texts) {
    return new Promise((resolve, reject) => {
        const p = spawn(PYTHON, [path.join(dir, "ml_predict.py")], { cwd: dir });
        let out = "", err = "";
        p.stdout.on("data", d => (out += d));
        p.stderr.on("data", d => (err += d));
        p.on("error", reject);
        p.on("close", code => {
            if (code !== 0) return reject(new Error(err.trim().split("\n").pop() || `exit ${code}`));
            try { resolve(JSON.parse(out)); } catch (e) { reject(e); }
        });
        p.stdin.on("error", () => {});
        p.stdin.end(JSON.stringify(texts));
    });
}

// รวมผลกฎกับโมเดล: spam ถ้าโมเดลให้คะแนน >= ML_THRESHOLD หรือกฎจับได้ชัดเจน (score >= SPAM_THRESHOLD)
async function classify(items) {
    let probs = null;
    try {
        probs = await mlScores(items.map(i => ({ text: i.text, dup: i.dup, authorDup: i.authorDup })));
    } catch (e) {
        console.log("ML ใช้ไม่ได้ ใช้กฎอย่างเดียว:", e.message);
    }
    return items.map((it, k) => {
        if (!probs) return { ...it, ml: null, label: (it.score >= 1 || it.lowQuality) ? "spam" : "normal" };
        const ml = probs[k];
        const spam = ml >= ML_THRESHOLD || it.score >= SPAM_THRESHOLD;
        return { ...it, ml, label: spam ? "spam" : "normal" };
    });
}

function readBody(req, limit = 10_000) {
    return new Promise((resolve, reject) => {
        let size = 0; const chunks = [];
        req.on("data", c => {
            size += c.length;
            if (size > limit) { reject(new Error("body too large")); req.destroy(); return; }
            chunks.push(c);
        });
        req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
        req.on("error", reject);
    });
}

const sendJson = (res, status, obj) => {
    res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
    res.end(JSON.stringify(obj));
};

const server = http.createServer(async (req, res) => {
    try {
        if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
            const html = loadIndex();
            if (!html) {
                res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
                return res.end(`ไม่พบ index.html ต้องวางไว้ที่ ${INDEX_PATHS[0]}\n(หรือข้างๆ server.js)`);
            }
            res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
            return res.end(html);
        }

        if (req.method === "POST" && req.url === "/api/analyze") {
            let body;
            try { body = JSON.parse(await readBody(req)); }
            catch { return sendJson(res, 400, { error: "ข้อมูลที่ส่งมาไม่ถูกต้อง" }); }

            const videoId = getVideoId(body.url);
            if (!videoId) return sendJson(res, 400, { error: "ลิงก์ไม่ใช่ลิงก์วิดีโอ YouTube (เช่น youtube.com/watch?v=… หรือ youtu.be/…)" });

            const max = Math.min(Math.max(Number(body.max) || DEFAULT_MAX, 100), HARD_MAX);
            const video = await fetchVideoInfo(videoId);
            const comments = await fetchComments(videoId, max);
            const items = await classify(analyze(comments));
            return sendJson(res, 200, { video, fetched: items.length, mlUsed: items.some(i => i.ml !== null), items });
        }

        return sendJson(res, 404, { error: "ไม่พบเส้นทางนี้" });
    } catch (err) {
        const [status, error] = mapError(err);
        console.log("analyze error:", err.response?.status ?? "", err.message); // ไม่ log key
        if (res.headersSent) return res.end();
        sendJson(res, status, { error });
    }
});

server.on("error", err => {
    if (err.code === "EADDRINUSE") {
        console.log(`พอร์ต ${PORT} ถูกใช้อยู่แล้ว ลองพอร์ตอื่น เช่น  PORT=${PORT + 1} npm start`);
        console.log(`หรือดูว่าโปรแกรมไหนใช้อยู่:  lsof -nP -iTCP:${PORT} -sTCP:LISTEN`);
        process.exit(1);
    }
    throw err;
});

server.listen(PORT, () => {
    console.log(`เปิดที่ http://localhost:${PORT}`);
    if (!INDEX_PATHS.some(fs.existsSync)) {
        console.log(`คำเตือน: ไม่พบ index.html ที่ ${INDEX_PATHS[0]} (หรือข้างๆ server.js)`);
    }
});
