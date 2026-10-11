import re, time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import config
from kb_data import KnowledgeBase

SYSTEM_RAG = """Bạn là Trợ lý SOC hỗ trợ SOC Analyst (Tier 1) phân tích Sysmon Logs.
QUY TẮC:
1. Chỉ dùng thông tin trong phần NGỮ CẢNH TRI THỨC và nội dung log do người dùng cung cấp.
2. Nếu ngữ cảnh không đủ, nói rõ "Không đủ thông tin trong cơ sở tri thức". TUYỆT ĐỐI không bịa Event ID, mã MITRE ATT&CK, CVE hay tên tệp.
3. Trích dẫn nguồn bằng [1], [2]...; chỉ nêu mã kỹ thuật MITRE xuất hiện trong ngữ cảnh.
4. Trả lời bằng tiếng Việt: **Tóm tắt hành vi**, **Giải thích kỹ thuật**, **Ánh xạ MITRE ATT&CK**, **Mức độ nghi ngờ** và **Gợi ý điều tra tiếp**.
Bạn chỉ hỗ trợ phân tích; không đề xuất hành động tự động chặn/cách ly."""
SYSTEM_BASE = """Bạn là Trợ lý SOC hỗ trợ SOC Analyst (Tier 1) phân tích Sysmon Logs.
Trả lời bằng tiếng Việt, gồm: Tóm tắt hành vi, Giải thích kỹ thuật, Ánh xạ MITRE ATT&CK, Mức độ nghi ngờ."""
_THINK = re.compile(r"<think>.*?</think>", re.S)


class SOCAssistant:
    def __init__(self, top_k=config.TOP_K, model=config.LLM_MODEL, kb=None):
        self.top_k = top_k
        self.kb = kb or KnowledgeBase()
        self.llm = ChatOllama(model=model, base_url=config.OLLAMA_URL, temperature=0.0)

    def answer(self, query, use_rag=True):
        t0 = time.perf_counter()
        hits = self.kb.search(query, self.top_k) if use_rag else []
        t1 = time.perf_counter()
        if use_rag:
            ctx = "\n\n".join(f"[{i}] ({h['type']} | {h['ref']} - {h['title']})\n{h['text']}" for i, h in enumerate(hits, 1))
            msgs = [SystemMessage(SYSTEM_RAG), HumanMessage(f"NGỮ CẢNH TRI THỨC:\n{ctx}\n\nSYSMON LOG / CÂU HỎI:\n{query}")]
        else:
            msgs = [SystemMessage(SYSTEM_BASE), HumanMessage(f"SYSMON LOG / CÂU HỎI:\n{query}")]
        out = self.llm.invoke(msgs).content
        t2 = time.perf_counter()
        return {"answer": _THINK.sub("", out).strip(), "contexts": [h["text"] for h in hits],
                "hits": hits, "retrieval_s": t1 - t0, "generation_s": t2 - t1, "total_s": t2 - t0}
