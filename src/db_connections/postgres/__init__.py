import os
import sys
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from src.logger import logging
from src.exception.exception_handler import AppException

POSTGRES_URI = os.environ["NEON_DATABASE_URL"]

try:
    pool = ConnectionPool(
        conninfo=POSTGRES_URI,
        min_size=2,
        max_size=20,
        kwargs={"autocommit": True},
        open=True,
    )

    checkpointer = PostgresSaver(pool)              #type: ignore
    checkpointer.setup()

    logging.info("Postgres checkpointer initialized and setup completed.")\
    
except Exception as e:
    logging.error(f"Error initializing Postgres checkpointer: {e}")
    raise AppException(e, sys)

def get_checkpointer() -> PostgresSaver:
    """
    Returns the checkpointer object.

    Returns:
        PostgresSaver: The checkpointer object.
    """
    return checkpointer