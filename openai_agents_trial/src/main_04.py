# _/_/_/ 質問内容から必要な語句を取り出すエージェント
import asyncio
import os

from agents import Agent, Runner, function_tool, set_tracing_export_api_key

import src.access_key as access_key

os.environ["OPENAI_API_KEY"] = access_key.OPENAI_API_KEY
set_tracing_export_api_key(access_key.OPENAI_API_KEY)


@function_tool
def get_country(country: str) -> str:
    return f"{country}の首都は東京です。"


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_country],
)


async def main() -> None:
    result = await Runner.run(agent, input="日本の首都は何ですか")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
