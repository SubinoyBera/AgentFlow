import os
import sys
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.logger import logging
from src.exception.exception_handler import AppException

POSTGRES_URI = os.environ["NEON_POSTGRES_URL"]

pool: AsyncConnectionPool | None = None
checkpointer: AsyncPostgresSaver | None = None


async def init_checkpointer() -> AsyncPostgresSaver:
    """
    Initializes the async Postgres connection pool and checkpointer.
    Must be called once, inside a running event loop, before get_checkpointer() is used.

    Returns:
        AsyncPostgresSaver: The initialized checkpointer object.
    """
    global pool, checkpointer

    try:
        pool = AsyncConnectionPool(
            conninfo=POSTGRES_URI,
            min_size=2,
            max_size=20,
            kwargs={"autocommit": True},
            open=False,  # open explicitly below, since __init__ can't await
        )
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)  # type: ignore
        await checkpointer.setup()

        logging.info("Async Postgres checkpointer initialized and setup completed.")
        return checkpointer

    except Exception as e:
        logging.error(f"Error initializing async Postgres checkpointer: {e}")
        raise AppException(e, sys)


def get_checkpointer() -> AsyncPostgresSaver:
    """
    Returns the checkpointer object. init_checkpointer() must have been
    awaited already, or this will raise.

    Returns:
        AsyncPostgresSaver: The checkpointer object.
    """
    if checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialized. Call `await init_checkpointer()` "
            "first, inside an async context, before using get_checkpointer()."
        )
    return checkpointer


async def close_checkpointer() -> None:
    """Closes the connection pool. Call on app shutdown."""
    if pool is not None:
        await pool.close()
        logging.info("Async Postgres connection pool closed.")