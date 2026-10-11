import streamlit as st
from rag import SOCAssistant

st.set_page_config(page_title="Intelligent SOC Assistant", layout="wide")
st.title("🛡️ Intelligent SOC Assistant (LLM + RAG · Sysmon + MITRE)")
with st.sidebar:
    use_rag = st.toggle("Bật RAG (tắt = Baseline)", True)
    top_k = st.selectbox("Top-K", [1, 3, 5], index=1)

@st.cache_resource
def load(top_k):
    return SOCAssistant(top_k=top_k)

query = st.text_area("Dán Sysmon Log hoặc đặt câu hỏi:", height=220)
if st.button("Phân tích") and query.strip():
    with st.spinner("Đang phân tích..."):
        r = load(top_k).answer(query, use_rag=use_rag)
    st.markdown(r["answer"])
    a, b, c = st.columns(3)
    a.metric("Retrieval (s)", f"{r['retrieval_s']:.2f}"); b.metric("Generation (s)", f"{r['generation_s']:.2f}"); c.metric("Tổng (s)", f"{r['total_s']:.2f}")
    for i, h in enumerate(r["hits"], 1):
        with st.expander(f"[{i}] {h['ref']} · {h['title']} · cos={h['score']:.3f}"):
            st.write(h["text"])
