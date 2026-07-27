import os
from functools import lru_cache

from langfuse import get_client
from langfuse.langchain import CallbackHandler


def is_langfuse_enabled() -> bool:
    """Indica si hay credenciales mínimas para exportar trazas y scores."""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    )


@lru_cache(maxsize=1)
def get_langchain_callbacks() -> list:
    """Retorna callbacks reutilizables de Langfuse cuando la integración está activa."""
    if not is_langfuse_enabled():
        return []

    try:
        return [CallbackHandler()]
    except Exception:
        return []


def get_langfuse_client():
    """Expone el cliente singleton inicializado por la librería."""
    return get_client()
