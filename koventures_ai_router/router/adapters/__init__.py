"""Provider adapters for the AI router."""

from .deepseek_v4 import DeepSeekV4Client
from .glm_5_2 import GLM52Client
from .hf_hosted import HFHostedAdapter
from .minimax_m3 import MiniMaxM3Client

__all__ = [
    "HFHostedAdapter",
    "GLM52Client",
    "DeepSeekV4Client",
    "MiniMaxM3Client",
]
