"""Run the RAG caching benchmark with GPT-5-mini as the main agent.

This wrapper intentionally bypasses ``src/benchmark_rag.py``'s ``main()``
selection so presentation runs are not affected by temporary model switches.
"""

import asyncio
import os
import sys


os.environ["OPTIMIZATION_METHOD"] = "caching"
os.environ.setdefault("CACHE_MODEL", "gpt-4o-mini")
os.environ.setdefault("CACHE_HIT_MODEL", os.environ["CACHE_MODEL"])
os.environ.setdefault("CACHE_MATCH_POLICY", "deterministic_gate")

sys.path.insert(0, "src")

from langchain_openai import ChatOpenAI  # noqa: E402

import benchmark_rag as benchmark  # noqa: E402


async def main() -> None:
    try:
        model_name = "gpt-5-mini"
        chat_model = ChatOpenAI(model=model_name, reasoning_effort="medium")
        await benchmark.process_benchmark(model_name=model_name, chat_model=chat_model)
    finally:
        await benchmark.es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
