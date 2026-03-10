"""Configuration constants for the Retrieval-Augmented Document QA app."""

APP_TITLE = "Retrieval-Augmented Document QA"
APP_DESCRIPTION = (
    "Upload documents, build a FAISS index, retrieve relevant chunks, and get grounded "
    "answers from Gemini."
)

MODEL_ID = "gemini-3.1-flash-lite-preview"
MODEL_OPTIONS = [
    "gemini-3.1-flash-lite-preview",
]
GENERATE_CONTENT_API = "streamGenerateContent"
TIMEOUT_SECONDS = 35

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_FILES = 3
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_PDF_PAGES = 40
MAX_CHARS_PER_FILE = 150_000
MAX_TOTAL_CHARS = 250_000

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
MAX_TOP_K = 5

EXAMPLE_QUESTIONS = [
    "What are the key objectives described in these documents?",
    "Summarize the most important requirements for a manager.",
    "List the main risks, constraints, or limitations mentioned.",
]
