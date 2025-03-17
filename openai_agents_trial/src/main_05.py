import asyncio
import os

from agents import Agent, Runner, function_tool, set_tracing_export_api_key

import src.access_key as access_key

os.environ["OPENAI_API_KEY"] = access_key.OPENAI_API_KEY
set_tracing_export_api_key(access_key.OPENAI_API_KEY)


@function_tool
def get_weather(city: str) -> str:
    return f"The weater in {city} is sunny."


@function_tool
def get_capital(country: str) -> str:
    capitals = {
        "Japan": "Tokyo",
        "USA": "Washington D.C.",
        "France": "Paris",
        "Germany": "Berlin",
    }
    return f"The capital of {country} is {capitals.get(country, 'unknown')}."


@function_tool
def translate(text: str, target_language: str) -> str:
    translations = {
        "Hello": {"Japanese": "こんにちは", "Spanish": "Hola"},
        "Goodbye": {"Japanese": "さようなら", "Spanish": "Adiós"},
    }
    return translations.get(text, {}).get(target_language, "Translation not found")


agent = Agent(
    name="Multi-tool Agent",
    instructions="You are a helpful agent",
    tools=[get_weather, get_capital, translate],
)


async def main() -> None:
    tasks = [
        Runner.run(agent, input="What is the weather in Tokyo?"),
        Runner.run(agent, input="What is the capital of Japan?"),
        Runner.run(agent, input="Translate Hello to Japanese"),
    ]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
