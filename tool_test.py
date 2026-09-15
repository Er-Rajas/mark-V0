from ollama import chat


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "What is the weather in Mumbai?"
        }
    ],
    tools=[get_weather]
)

print("CONTENT:")
print(response.message.content)

print("\nTOOL CALLS:")
print(response.message.tool_calls)