import logging


from app.config import PROVIDER_CHAIN
from app.providers.client import call_provider

logger = logging.getLogger("llm-router")


class AllProvidersFailedError(Exception):
    pass


def generate_with_fallback(prompt: str) -> dict:
    errors = []

    for provider in PROVIDER_CHAIN:
        if provider.requires_key and provider.api_key == "not-needed":
            logger.warning(f"Skip {provider.name}: API key not set")
            errors.append({"provider": provider.name, "error": "missing_api_key"})
            continue
    
        try:
            timeout = 180.0 if provider.name == "ollama" else 15.0
            text = call_provider(provider, prompt, timeout=timeout)
            return {
                "text": text,
                "provider_used": provider.name,
                "model_used": provider.model,
                "fallback_count": len(errors),
            }
        except Exception as exc:
            logger.warning(f"{provider.name} failed: {exc}")
            errors.append({"provider": provider.name, "error": str(exc)})
            continue

    raise AllProvidersFailedError(f"Semua provider gagal: {errors}")