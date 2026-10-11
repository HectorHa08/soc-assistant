import os
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")          # Ollama local; máy yếu: qwen3:1.7b
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "BAAI/bge-m3"                             # PHẢI trùng model đã tạo vector của nhóm
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
TOP_K = int(os.getenv("TOP_K", 3))
MAX_QUERY_CHARS = 2000
