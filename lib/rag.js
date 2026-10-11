// RAG core: bge-m3 query embedding (API) + exact cosine search over the team's prebuilt vectors + Qwen3 prompt.
const fs = require("fs");
const path = require("path");

const DIM = 1024;
const MAX_QUERY_CHARS = 2000;
let KB = null;

function loadKB() {
  if (KB) return KB;
  const dir = path.join(process.cwd(), "data");
  const buf = fs.readFileSync(path.join(dir, "vectors.f32"));
  const vec = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const chunks = JSON.parse(fs.readFileSync(path.join(dir, "chunks.json"), "utf-8"));
  if (vec.length !== chunks.length * DIM) throw new Error("vectors.f32 và chunks.json không khớp");
  KB = { vec, chunks };
  return KB;
}

async function embedQuery(text) {
  const base = process.env.EMBED_BASE_URL || "https://api.siliconflow.com/v1";
  const key = process.env.EMBED_API_KEY || process.env.LLM_API_KEY;
  const r = await fetch(`${base}/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: process.env.EMBED_MODEL || "BAAI/bge-m3",
      input: text.slice(0, MAX_QUERY_CHARS),
      encoding_format: "float",
    }),
  });
  if (!r.ok) throw new Error(`Embedding API ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const q = Float32Array.from((await r.json()).data[0].embedding);
  let n = 0;
  for (let i = 0; i < q.length; i++) n += q[i] * q[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < q.length; i++) q[i] /= n;
  return q;
}

// Inner product trên vector đã chuẩn hóa = cosine; tương đương faiss.IndexFlatIP của nhóm.
function search(q, k) {
  const { vec, chunks } = loadKB();
  const n = chunks.length;
  const top = []; // [{i, s}] giữ k phần tử điểm cao nhất
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * DIM;
    for (let j = 0; j < DIM; j++) s += vec[o + j] * q[j];
    if (top.length < k) {
      top.push({ i, s });
      top.sort((a, b) => b.s - a.s);
    } else if (s > top[k - 1].s) {
      top[k - 1] = { i, s };
      top.sort((a, b) => b.s - a.s);
    }
  }
  return top.map(({ i, s }) => ({ ...chunks[i], score: s }));
}

const SYSTEM_RAG = `Bạn là Trợ lý SOC hỗ trợ SOC Analyst (Tier 1) phân tích Sysmon Logs.
QUY TẮC:
1. Chỉ dùng thông tin trong phần NGỮ CẢNH TRI THỨC và nội dung log do người dùng cung cấp.
2. Nếu ngữ cảnh không đủ, nói rõ "Không đủ thông tin trong cơ sở tri thức". TUYỆT ĐỐI không bịa Event ID, mã MITRE ATT&CK, CVE hay tên tệp.
3. Trích dẫn nguồn bằng [1], [2]... tương ứng đoạn ngữ cảnh; chỉ nêu mã kỹ thuật MITRE xuất hiện trong ngữ cảnh.
4. Trả lời bằng tiếng Việt, theo cấu trúc:
- **Tóm tắt hành vi**
- **Giải thích kỹ thuật** (Event ID, các trường quan trọng, câu lệnh/obfuscation)
- **Ánh xạ MITRE ATT&CK** (mã kỹ thuật + lý do)
- **Mức độ nghi ngờ** (Thấp/Trung bình/Cao) và **Gợi ý điều tra tiếp**
Bạn chỉ hỗ trợ phân tích; không đề xuất hành động tự động chặn/cách ly.`;

const SYSTEM_BASE = `Bạn là Trợ lý SOC hỗ trợ SOC Analyst (Tier 1) phân tích Sysmon Logs.
Trả lời bằng tiếng Việt, gồm: Tóm tắt hành vi, Giải thích kỹ thuật, Ánh xạ MITRE ATT&CK, Mức độ nghi ngờ.`;

function buildMessages(query, hits) {
  if (!hits) {
    return [
      { role: "system", content: SYSTEM_BASE },
      { role: "user", content: `SYSMON LOG / CÂU HỎI:\n${query}` },
    ];
  }
  const ctx = hits
    .map((h, i) => `[${i + 1}] (${h.type} | ${h.ref} - ${h.title})\n${h.text}`)
    .join("\n\n");
  return [
    { role: "system", content: SYSTEM_RAG },
    { role: "user", content: `NGỮ CẢNH TRI THỨC:\n${ctx}\n\nSYSMON LOG / CÂU HỎI:\n${query}` },
  ];
}

// Gọi LLM (OpenAI-compatible, mặc định SiliconFlow + Qwen3-8B), stream từng token qua onToken.
async function streamLLM(messages, onToken) {
  const base = process.env.LLM_BASE_URL || "https://api.siliconflow.com/v1";
  const r = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.LLM_API_KEY}` },
    body: JSON.stringify({
      model: process.env.LLM_MODEL || "Qwen/Qwen3-8B",
      messages,
      stream: true,
      temperature: 0,
      max_tokens: 1400,
      enable_thinking: false, // Qwen3: tắt chế độ thinking để giảm độ trễ
    }),
  });
  if (!r.ok) throw new Error(`LLM API ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const dec = new TextDecoder();
  let buf = "";
  for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const t = JSON.parse(data).choices?.[0]?.delta?.content;
        if (t) onToken(t);
      } catch (_) {}
    }
  }
}

module.exports = { loadKB, embedQuery, search, buildMessages, streamLLM, MAX_QUERY_CHARS };
