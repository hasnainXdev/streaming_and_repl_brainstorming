import asyncio
from agents import Agent, run_demo_loop, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


async def main() -> None:
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
    )
    await run_demo_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
