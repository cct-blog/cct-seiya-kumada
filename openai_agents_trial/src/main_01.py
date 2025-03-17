
# _/_/_/ 何でも答えてくれるアシスタント

import os

from agents import Agent, Runner, set_tracing_export_api_key

import src.access_key as access_key

os.environ["OPENAI_API_KEY"] = access_key.OPENAI_API_KEY
set_tracing_export_api_key(access_key.OPENAI_API_KEY)

if __name__ == "__main__":
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    result = Runner.run_sync(agent, "川端康成の有名な小説の中から一つ選び、その書き出しを教えてください。")
    print(result.final_output)
