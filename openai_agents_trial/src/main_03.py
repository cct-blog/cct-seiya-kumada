# _/_/_/ 適切なエージェントにリクエストを転送するエージェント（並列処理版）
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


def hoge(n: int) -> int:
    return n


async def handle_request(agent, message) -> None:
    reslt = await Runner.run(agent, message)
    print(reslt.final_output)


async def main() -> None:
    # 複数のリクエストを並列に処理する
    # 日本語は日本語エージェントに、スペイン語はスペイン語エージェントに、
    # 英語は英語エージェントに転送される。
    tasks = [
        handle_request(TRIAGE_AGENT, "こんにちは、お元気ですか"),
        handle_request(TRIAGE_AGENT, "Hola, ¿cómo estás?"),
        handle_request(TRIAGE_AGENT, "Hello, how are you?"),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
