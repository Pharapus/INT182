// ดึงคอมเมนต์จาก YouTube ต่อท้ายไฟล์ data/comments.jsonl
// ใช้:  npm run collect     (ต้องตั้ง YT_API_KEY ในไฟล์ .env ก่อน)
import axios from "axios";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline/promises";
import { fileURLToPath } from "node:url";
import { stdin as input, stdout as output } from "node:process";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
try { process.loadEnvFile(path.join(root, ".env")); } catch { /* ไม่มี .env ก็ข้าม */ }

const API_KEY = process.env.YT_API_KEY;
const OUT_FILE = path.join(root, "data", "comments.jsonl");
const MAX_COMMENTS = Number(process.env.MAX_COMMENTS || 500); // ต่อวิดีโอ

if (!API_KEY) {
    console.log("ยังไม่ได้ตั้งค่า YT_API_KEY ให้คัดลอก .env.example เป็น .env แล้วใส่คีย์ (อย่าเขียนคีย์ลงในโค้ด)");
    process.exit(1);
}

const rl = readline.createInterface({ input, output });
const videoUrl = await rl.question("ใส่ YouTube URL: ");
rl.close();

function getVideoId(raw) {
    const url = new URL(raw.trim());
    if (url.hostname === "youtu.be") return url.pathname.slice(1);
    return url.searchParams.get("v");
}

// อ่าน commentId ที่เคยเก็บแล้ว จะได้ไม่เก็บซ้ำ
function loadExistingIds() {
    const ids = new Set();
    if (!fs.existsSync(OUT_FILE)) return ids;
    for (const line of fs.readFileSync(OUT_FILE, "utf8").split("\n")) {
        if (!line.trim()) continue;
        try { const r = JSON.parse(line); if (r.commentId) ids.add(r.commentId); } catch {}
    }
    return ids;
}

try {
    const videoId = getVideoId(videoUrl);
    if (!videoId) throw new Error("ไม่พบ Video ID ใน URL");
    console.log("\nกำลังดึง comments...\n");

    const fetched = [];
    let pageToken;
    do {
        const response = await axios.get("https://www.googleapis.com/youtube/v3/commentThreads", {
            params: { part: "snippet", videoId, maxResults: 100, pageToken, textFormat: "plainText", key: API_KEY }
        });
        for (const item of response.data.items) {
            const top = item.snippet.topLevelComment;
            const s = top.snippet;
            fetched.push({
                videoId,
                commentId: top.id,
                authorChannelId: s.authorChannelId?.value ?? null,
                text: s.textOriginal ?? s.textDisplay,
                likeCount: s.likeCount,
                replyCount: item.snippet.totalReplyCount,
                publishedAt: s.publishedAt
            });
        }
        pageToken = response.data.nextPageToken;
    } while (pageToken && fetched.length < MAX_COMMENTS);

    const existing = loadExistingIds();
    const fresh = fetched.slice(0, MAX_COMMENTS).filter(c => !existing.has(c.commentId));
    if (fresh.length === 0) { console.log("ไม่มี comments ใหม่ (ไม่พบ หรือเคยเก็บครบแล้ว)"); process.exit(0); }

    fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
    fs.appendFileSync(OUT_FILE, fresh.map(c => JSON.stringify(c)).join("\n") + "\n", "utf8");
    console.log("ดึง comments สำเร็จ!");
    console.log(`Video ID: ${videoId}`);
    console.log(`ดึงมา ${fetched.length} | เพิ่มใหม่ ${fresh.length} | ข้ามที่ซ้ำ ${fetched.length - fresh.length}`);
} catch (error) {
    if (error.response) {
        const message = error.response.data?.error?.message;
        console.log("YouTube API Error:", error.response.status, message);
        if (message && message.includes("disabled comments")) console.log("❌ วิดีโอนี้ปิดการแสดงความคิดเห็น");
    } else {
        console.log("Error:", error.message);
    }
}
