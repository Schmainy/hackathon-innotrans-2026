"""Re-Export von A's Azure-Client (liegt unveraendert in src/tools/llm_client.py, wo A's Code ihn importiert)."""
from src.tools.llm_client import DEFAULT_TIMEOUT, LLMConfigError, LLMError, call_llm  # noqa: F401
