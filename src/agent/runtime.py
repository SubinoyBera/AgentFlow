import asyncio
import time
import json
import threading
import queue
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Optional, Iterator, Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamMode
from src.logger import logging

_MCP_CONFIG_PATH = Path("D:/My Projects/Multi-AI Agent/mcp_servers/config.json")


class AgentRuntime:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._exit_stack: Optional[AsyncExitStack] = None

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=250):
            raise RuntimeError("AgentRuntime failed to start within 250s -- check logs for MCP session errors.")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._startup())
        self._loop.run_forever()

    async def _startup(self):
        import src.agent.workflow as workflow

        with open(_MCP_CONFIG_PATH) as f:
            server_config = json.load(f)

        self._exit_stack = AsyncExitStack()
        client = MultiServerMCPClient(server_config)

        async def _open(server_name: str):
            try:
                session = await self._exit_stack.enter_async_context(client.session(server_name))                   #type: ignore
                tools = await load_mcp_tools(session)
                logging.info(f"MCP session '{server_name}' ready.")
                return server_name, tools
            except Exception as e:
                logging.error(f"Failed to open MCP session for '{server_name}': {e}", exc_info=True)
                return server_name, []

        results = await asyncio.gather(*(_open(name) for name in server_config))

        for server_name, tools in results:
            if server_name in ("gmail", "calendar"):
                if tools:
                    workflow.n8n_agents[server_name] = tools[0]
            else:
                workflow.mcp_tools.extend(tools)
        
        await workflow.build_agent() 
        self._graph = workflow.ai_agent
        self._ready.set()

    def invoke(self, payload, config: RunnableConfig):
        fut = asyncio.run_coroutine_threadsafe(self._graph.ainvoke(payload, config=config), self._loop)
        return fut.result()

    def get_state(self, config: RunnableConfig):
        fut = asyncio.run_coroutine_threadsafe(self._graph.aget_state(config), self._loop)                         
        return fut.result()

    def shutdown(self):
        if self._exit_stack is not None:
            fut = asyncio.run_coroutine_threadsafe(self._exit_stack.aclose(), self._loop)
            try:
                fut.result(timeout=10)
            except Exception as e:
                logging.error(f"Error during MCP session shutdown: {e}", exc_info=True)

            from src.db_connections.postgres import close_checkpointer
            fut2 = asyncio.run_coroutine_threadsafe(close_checkpointer(), self._loop)
            try:
                fut2.result(timeout=10)
            except Exception as e:
                logging.error(f"Error during checkpointer pool shutdown: {e}", exc_info=True)
        
        self._loop.call_soon_threadsafe(self._loop.stop)

    def stream(self, payload, config: RunnableConfig, stream_mode: StreamMode | list[StreamMode]) -> Iterator[dict[str, Any]]:
        """
        Sync generator -- use with a normal `for chunk in runtime.stream(...)`.
        The actual streaming runs on the background loop (so it reuses the
        persistent MCP sessions); each chunk is relayed to the calling
        (Streamlit) thread through a thread-safe queue as soon as it's ready.
        """
        q: queue.Queue = queue.Queue()
        _DONE = object()

        async def _pump():
            try:
                async for chunk in self._graph.astream(payload, config=config, stream_mode=stream_mode):
                    q.put(chunk)
            except Exception as e:
                q.put(("__error__", e))
            finally:
                q.put(_DONE)

        asyncio.run_coroutine_threadsafe(_pump(), self._loop)

        while True:
            item = q.get()  # blocks the CALLING thread only, not the background loop
            if item is _DONE:
                return
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                raise item[1]
            yield item                              #type: ignore

    def retrieve_all_threads(self):
        """
        Async-safe version of retrieve_all_threads(checkpointer), run on the
        background loop and returned synchronously for Streamlit's thread.
        """
        async def _retrieve_all_threads():
            checkpointer = self._graph.checkpointer
            all_threads = set()
            async for checkpoint in checkpointer.alist(None):
                configurable = checkpoint.config.get("configurable")
                if configurable and "thread_id" in configurable:
                    all_threads.add(configurable["thread_id"])
            return list(all_threads)

        fut = asyncio.run_coroutine_threadsafe(_retrieve_all_threads(), self._loop)
        return fut.result()


_runtime: Optional[AgentRuntime] = None
_runtime_lock = threading.Lock()

def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = AgentRuntime()
    return _runtime