import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str
    requires_key: bool = True

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "not-needed")


# Urutan = urutan fallback. Diubah tanpa sentuh logic router.
PROVIDER_CHAIN: list[ProviderConfig] = [
    ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        model="gemini-3.6-flash",
    ),
    ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.3-70b-versatile",
    ),
    ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat",
    ),
    ProviderConfig(
        name="ollama",
        base_url="http://ollama:11434/v1",
        api_key_env="OLLAMA_API_KEY",  # tidak dipakai beneran, hanya syarat SDK
        model="llama3.2",
        requires_key=False,
    )
]