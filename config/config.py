# # config/config.py
# import os
# from pathlib import Path
# from dotenv import load_dotenv

# load_dotenv()

# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# # Gemini/Google GenAI API key (from AI Studio)
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# # Models (change if you want other Gemini variants)
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
# LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-2.5")  # change if you want flash / other

# # Vector store paths
# VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(DATA_DIR / "faiss.index"))
# METADATA_PATH = os.getenv("METADATA_PATH", str(DATA_DIR / "metadata.json"))

# # Chunking params (words)
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# MAX_RETRIEVALS = int(os.getenv("MAX_RETRIEVALS", "8"))  # increased to allow more context

# # Retrieval
# MAX_RETRIEVALS = int(os.getenv("MAX_RETRIEVALS", "6"))

# # Web fallback (not recommended for private/internal policies)
# ALLOW_WEB_FALLBACK = os.getenv("ALLOW_WEB_FALLBACK", "false").lower() == "true"
# SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
















# config/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# BASE DIRECTORIES
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# API KEYS
# ----------------------------

# Groq API key (used for BOTH LLM and embeddings)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Google API key ONLY IF you still use Gemini anywhere else
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ----------------------------
# MODEL CONFIGS
# ----------------------------

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM model: switched from Gemini Flash → Groq LLaMA 3.1
# Available Groq models include:
# - llama-3.1-8b-instant
# - llama-3.1-70b-versatile
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ----------------------------
# VECTOR STORE / FILE PATHS
# ----------------------------
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(DATA_DIR / "faiss.index"))
METADATA_PATH = os.getenv("METADATA_PATH", str(DATA_DIR / "metadata.json"))

# ----------------------------
# CHUNKING PARAMETERS
# CHUNK_SIZE & CHUNK_OVERLAP are CHARACTER lengths
# ----------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))         # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))    # characters

# ----------------------------
# RETRIEVAL CONFIG
# ----------------------------
MAX_RETRIEVALS = int(os.getenv("MAX_RETRIEVALS", "8"))

# ----------------------------
# WEB SEARCH FALLBACK (optional)
# ----------------------------
ALLOW_WEB_FALLBACK = os.getenv("ALLOW_WEB_FALLBACK", "false").lower() == "true"
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

