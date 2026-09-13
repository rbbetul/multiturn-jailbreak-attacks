"""Simple test: verify Ollama gemma4 responds."""

from ollama import chat

response = chat(
    model="gemma4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.message.content)
