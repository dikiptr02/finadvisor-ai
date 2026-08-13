from app.config import ProviderConfig
from openai import OpenAI


def call_provider(provider: ProviderConfig, prompt: str, timeout: float = 15.0) -> str:
    client = OpenAI(base_url=provider.base_url, api_key=provider.api_key, timeout=timeout)

    response = client.chat.completions.create(
        model=provider.model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content