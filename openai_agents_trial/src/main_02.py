# _/_/_/ 適切なエージェントにリクエストを転送するエージェント

import asyncio
import os

from agents import Agent, Runner, set_tracing_export_api_key

import src.access_key as access_key

os.environ["OPENAI_API_KEY"] = access_key.OPENAI_API_KEY
set_tracing_export_api_key(access_key.OPENAI_API_KEY)

SPANISH_AGENT = Agent(name="Spanish agent", instructions="You only speak Spanish.")
JPANENSE_AGENT = Agent(name="Japanese agent", instructions="You only speak Japanese.")
ENGLISH_AGENT = Agent(name="English agent", instructions="You only speak English.")
TRIAGE_AGENT = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[SPANISH_AGENT, JPANENSE_AGENT, ENGLISH_AGENT],
)


async def main() -> None:
    # 日本語で聞けば日本語エージェントに転送される。
    # reslt = await Runner.run(TRIAGE_AGENT, "こんにちは、お元気ですか")
    # 英語で聞けば英語エージェントに転送される。
    # reslt = await Runner.run(TRIAGE_AGENT, "Hello, Hou are you?")
    # スペイン語で聞けばスペイン語エージェントに転送される。
    reslt = await Runner.run(TRIAGE_AGENT, "Hola, ¿cómo estás?")
    print(reslt.final_output)


if __name__ == "__main__":
    asyncio.run(main())
