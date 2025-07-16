import asyncio
import random
from agents import (
    Agent,
    ItemHelpers,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    RunConfig,
)
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# set_tracing_disabled(disabled=True)

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


# @function_tool
# def how_many_jokes() -> int:
#     return random.randint(1, 10)



async def main():
    agent = Agent(
        name="Assistent",
        instructions="You are a assistant!",
    )

    result = Runner.run_streamed(
        agent,
        input="Hello",
        run_config=config,
        max_turns=2
    )
    print("=== Run starting ===")

    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(
                    f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                )
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")


if __name__ == "__main__":
    asyncio.run(main())
