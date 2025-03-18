# _/_/_/ 今日の天気予報を教えてくれるエージェント
import asyncio
import os

import requests
from agents import Agent, Runner, function_tool, set_tracing_export_api_key
from deep_translator import GoogleTranslator

import src.access_key as access_key

os.environ["OPENAI_API_KEY"] = access_key.OPENAI_API_KEY
set_tracing_export_api_key(access_key.OPENAI_API_KEY)
OPENWEATHERMAP_API_KEY = access_key.OPENWEATHERMAP_API_KEY


def translate_text(text: str) -> str:
    translator = GoogleTranslator(source="auto", target="en")
    result = translator.translate(text)
    return result


@function_tool
def get_weather(city: str) -> str:
    eng_city = translate_text(city)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={eng_city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        temp_min = data["main"]["temp_min"]
        temp_max = data["main"]["temp_max"]
        humidity = data["main"]["humidity"]
        return f"The weather in {eng_city} is {weather_description} \
                with average temperature of {temp}°C, \
                minimum temperature of {temp_min}°C, \
                maximum temperature of {temp_max}°C, \
                and humidity of {humidity}%."
    else:
        return "Sorry, I couldn't retrieve the weather information."


agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent that provides weather information.",
    tools=[get_weather],
)


async def main() -> None:
    result = await Runner.run(agent, input="今日の東京の天気を教えてください。")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
