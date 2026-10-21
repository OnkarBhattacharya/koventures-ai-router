"""Provider adapters for the AI router."""

from .gemini_flash import GeminiFlashClient
from .openrouter import OpenRouterClient
from .ollama import OllamaClient
from .glm_5_2 import GLM52Client
from .deepseek_v4 import DeepSeekV4Client
from .minimax_m3 import MiniMaxM3Client

__all__ = [
    "GeminiFlashClient",
    "OpenRouterClient",
    "OllamaClient",
    "GLM52Client",
    "DeepSeekV4Client",
    "MiniMaxM3Client",
]
