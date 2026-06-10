"""Run the RAG benchmark with Gemini 2.5 Flash as the main agent."""

import asyncio
import os
import sys

from dotenv import load_dotenv


METHOD = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPTIMIZATION_METHOD", "none")).lower()
os.environ["OPTIMIZATION_METHOD"] = METHOD
os.environ.setdefault("MAIN_MODEL", "gemini-2.5-flash")

if METHOD == "caching":
    os.environ.setdefault("CACHE_MODEL", "gpt-4o-mini")
    os.environ.setdefault("CACHE_HIT_MODEL", os.environ["CACHE_MODEL"])
    os.environ.setdefault("CACHE_MATCH_POLICY", "deterministic_gate")

load_dotenv()

sys.path.insert(0, "src")

import benchmark_rag as benchmark  # noqa: E402


async def main() -> None:
    try:
        model_name = os.environ["MAIN_MODEL"]
        chat_model = benchmark.create_chat_model(model_name, temperature=0.0)
        await benchmark.process_benchmark(model_name=model_name, chat_model=chat_model)
    finally:
        await benchmark.es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
