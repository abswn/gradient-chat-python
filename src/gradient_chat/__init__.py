import importlib.metadata

from .client import GradientChatClient, GradientChatError
from .conversation import GradientConversation

__all__ = ["GradientChatClient", "GradientChatError", "GradientConversation"]

# Dynamic version
try:
    __version__ = importlib.metadata.version("gradient-chat")
except importlib.metadata.PackageNotFoundError:
    # Fallback for running directly from git clone
    __version__ = "Unknown"
