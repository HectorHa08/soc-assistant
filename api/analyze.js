// POST /api/analyze  {query, topK, useRag}  ->  NDJSON stream: meta | token | done | error
const { embedQuery, search, buildMessages, streamLLM } = require("../lib/rag");

module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const code = process.env.ACCESS_CODE; // tùy chọn: bảo vệ API key khỏi người lạ
  if (code && req.headers["x-access-code"] !== code) return res.status(401).json({ error: "Sai mã truy cập" });

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  const query = String(body.query || "").trim();
  const useRag = body.useRag !== false;
  const topK = Math.min(8, Math.max(1, parseInt(body.topK, 10) || 3));
  if (!query) return res.status(400).json({ error: "Thiếu nội dung query" });
  if (query.length > 8000) return res.status(400).json({ error: "Query quá dài (tối đa 8000 ký tự)" });
  if (!process.env.LLM_API_KEY) return res.status(500).json({ error: "Chưa cấu hình LLM_API_KEY" });

  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  const send = (o) => res.write(JSON.stringify(o) + "\n");

  const t0 = performance.now();
  try {
    let hits = null;
    let embedMs = 0, searchMs = 0;
    if (useRag) {
      const q = await embedQuery(query);
      const t1 = performance.now();
      hits = search(q, topK);
      const t2 = performance.now();
      embedMs = t1 - t0;
      searchMs = t2 - t1;
    }
    const tR = performance.now();
    send({
      type: "meta",
      useRag,
      topK,
      contexts: (hits || []).map((h) => ({ id: h.id, type: h.type, ref: h.ref, title: h.title, url: h.url, score: h.score, text: h.text })),
      retrieval_s: (tR - t0) / 1000,
      embed_s: embedMs / 1000,
      search_s: searchMs / 1000,
    });

    let first = null;
    await streamLLM(buildMessages(query, hits), (tok) => {
      if (first === null) first = performance.now();
      send({ type: "token", t: tok });
    });
    const tEnd = performance.now();
    send({
      type: "done",
      retrieval_s: (tR - t0) / 1000,
      ttft_s: first ? (first - tR) / 1000 : null,
      generation_s: (tEnd - tR) / 1000,
      total_s: (tEnd - t0) / 1000,
    });
  } catch (e) {
    send({ type: "error", message: String(e.message || e) });
  }
  res.end();
};
