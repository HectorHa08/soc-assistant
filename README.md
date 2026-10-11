# Intelligent SOC Assistant — LLM + RAG (Sysmon & MITRE ATT&CK)

Hệ thống nạp **dataset của nhóm** (KLTN_Knowledge_Base, `faiss_merged`): 4.308 chunk, vector **BAAI/bge-m3** (1024 chiều), gồm Sysmon Event ID và MITRE ATT&CK Enterprise (Overview / Detection / Mitigation).
Sau khi làm sạch còn **4.202 chunk** (bỏ 106 chunk "Detection of X:" rỗng, xóa mã `(Citation: ...)`, link markdown, lỗi font trong tài liệu Sysmon).

```
api/analyze.js   Vercel function: embed query (bge-m3) -> cosine top-K -> prompt -> Qwen3 (stream)
lib/rag.js       lõi RAG
public/index.html  giao diện web
data/            vectors.f32 (17 MB) + chunks.json + ground_truth.csv
scripts/export_data.py   tạo lại data/ từ thư mục KLTN_Knowledge_Base
local/           bản Python (Ollama + Qwen3 local): app Streamlit, evaluate.py, eval_retrieval.py
```

## Triển khai lên Vercel

Vercel không chạy được Ollama/Streamlit/torch, nên bản web gọi **API**: bge-m3 để nhúng câu hỏi (bắt buộc đúng model vì vector của nhóm tạo bằng bge-m3) và Qwen3 để sinh câu trả lời. Mặc định dùng SiliconFlow (OpenAI-compatible, có `BAAI/bge-m3` và `Qwen/Qwen3-8B`); nhà cung cấp khác tương thích cũng dùng được bằng cách đổi biến môi trường.

1. Tạo tài khoản + API key tại SiliconFlow (siliconflow.com).
2. Đưa thư mục này lên GitHub (repo private), rồi vào vercel.com → **Add New → Project → Import** repo. Framework chọn **Other**, không cần build command.
   Hoặc dùng CLI: `npm i -g vercel` → `vercel login` → `vercel` (preview) → `vercel --prod`.
3. Project → **Settings → Environment Variables** thêm (xem `.env.example`):
   `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBED_BASE_URL`, `EMBED_MODEL=BAAI/bge-m3`, và nên có `ACCESS_CODE` (mã bí mật nhóm nhập trên web) để người lạ không dùng hết hạn mức API.
4. **Redeploy** để biến môi trường có hiệu lực, mở URL `*.vercel.app` và thử các mẫu log.

Lưu ý: Hobby plan phù hợp đồ án; hàm đặt `maxDuration` 60 giây. Cold start lần đầu đọc 17 MB dữ liệu nên chậm hơn vài trăm ms.

## Chạy local bằng Python (Ollama)
```
cd local
python -m pip install -r requirements.txt
ollama pull qwen3:4b
python eval_retrieval.py          # Hit@1/3/5 + MRR trên ground truth của nhóm (RQ3)
streamlit run app.py
python evaluate.py compare        # Baseline vs LLM+RAG: P/R/F1, latency (RQ1, RQ2, RQ4)
python evaluate.py ablation       # Top-K 1/3/5
```
Lần đầu `sentence-transformers` tải bge-m3 (~2.2 GB).

## Cập nhật dataset
Đổi/bổ sung dữ liệu trong thư mục KLTN_Knowledge_Base rồi chạy:
`python scripts/export_data.py <đường_dẫn_KLTN_Knowledge_Base>` và commit lại thư mục `data/`.
