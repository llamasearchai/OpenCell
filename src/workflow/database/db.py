import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator
import asyncio

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import text
from sqlalchemy.exc import OperationalError, InterfaceError

from .models import Base

# Configure logging
logger = logging.getLogger(__name__)
# Ensure logging is configured. If you have a central logging setup, this might not be needed here.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Database URL: Prefer environment variable, fallback to opencell.db
# Ensure DATABASE_URL and ASYNC_DATABASE_URL are correctly defined or fall back gracefully.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///./opencell.db"  # Default to opencell.db for synchronous operations
)
ASYNC_DATABASE_URL = os.environ.get(
    "ASYNC_DATABASE_URL", "sqlite+aiosqlite:///./opencell.db"  # Default to opencell.db for asynchronous operations
)

# Connection pooling and engine configuration
ENGINE_OPTIONS = {
    "pool_size": int(os.environ.get("DB_POOL_SIZE", 10)),
    "max_overflow": int(os.environ.get("DB_MAX_OVERFLOW", 20)),
    "pool_timeout": int(os.environ.get("DB_POOL_TIMEOUT", 30)),
    "pool_recycle": int(os.environ.get("DB_POOL_RECYCLE", 1800)),
    "echo": os.environ.get("DB_ECHO", "False").lower() == "true",  # For debugging SQL queries
}

sync_engine = None
SyncSessionLocal = None
async_engine = None
AsyncSessionLocal = None

try:
    # Create synchronous engine
    if DATABASE_URL:
        sync_engine = create_engine(
            DATABASE_URL, **{k: v for k, v in ENGINE_OPTIONS.items() if k != "echo"}
        )  # echo not needed for sync typically
        SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
        logger.info(
            f"Successfully created synchronous database engine for: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}"
        )  # Hide credentials

    # Create asynchronous engine
    if ASYNC_DATABASE_URL:
        async_engine = create_async_engine(ASYNC_DATABASE_URL, **ENGINE_OPTIONS)
        AsyncSessionLocal = sessionmaker(
            bind=async_engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
        )
        logger.info(
            f"Successfully created asynchronous database engine for: {ASYNC_DATABASE_URL.split('@')[-1] if '@' in ASYNC_DATABASE_URL else ASYNC_DATABASE_URL}"
        )  # Hide credentials

except SQLAlchemyError as e:
    logger.error(f"Error creating database engine: {e}", exc_info=True)
    raise
except ImportError as e:
    logger.error(
        f"Database driver not found. Please install the required driver (e.g., psycopg2-binary for PostgreSQL, aiomysql for MySQL, aiosqlite for SQLite): {e}"
    )
    # Instructions to user could be added here
    raise


# Dependency to get a synchronous database session
@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    if not SyncSessionLocal:
        logger.error("Synchronous database session not configured.")
        raise RuntimeError("Synchronous database session not configured.")
    db = SyncSessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error in synchronous session: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


async def init_db_engines():
    """Initialize database engines with connection pooling"""
    global async_engine, sync_engine
    
    # Configure async engine with connection pooling
    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True
    )
    
    # Configure sync engine
    sync_engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True
    )
    
    logger.info("Database engines initialized with connection pooling")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions with enhanced error handling.
    
    Features:
    - Connection pooling
    - Automatic retry for transient errors
    - Connection health checks
    """
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        session = AsyncSession(bind=get_async_engine_instance(), expire_on_commit=False)
        try:
            # Test connection health
            await session.execute(text("SELECT 1"))
            
            yield session
            await session.commit()
            break
        except (OperationalError, InterfaceError) as e:
            await session.rollback()
            if attempt == max_retries - 1:
                raise DatabaseConnectionError(
                    f"Failed to establish database connection after {max_retries} attempts"
                ) from e
            logger.warning(
                f"Database connection failed (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s..."
            )
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()


async def init_db():  # Base is imported, no need to pass as arg
    if not async_engine:
        logger.error("Asynchronous engine not initialized. Cannot init_db.")
        return
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized (if they didn't exist).")


async def create_tables() -> None:
    """Create database tables. Alias for init_db for backward compatibility or specific use cases."""
    await init_db()


async def drop_tables() -> None:
    """Drop all database tables defined in Base.metadata. Use with caution!"""
    if not async_engine:
        logger.error("Asynchronous engine not initialized. Cannot drop_tables.")
        return
    logger.warning("Dropping all database tables! This is a destructive operation.")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All database tables dropped.")


async def close_db_engines():
    """Gracefully close database engine connections."""
    if sync_engine:
        sync_engine.dispose()
        logger.info("Synchronous database engine disposed.")
    if async_engine:
        await async_engine.dispose()
        logger.info("Asynchronous database engine disposed.")


# Functions to get engines if needed directly
def get_async_engine_instance():  # Renamed for clarity
    if not async_engine:
        raise RuntimeError("Asynchronous engine not configured.")
    return async_engine


def get_sync_engine_instance():  # Renamed for clarity
    if not sync_engine:
        raise RuntimeError("Synchronous engine not configured.")
    return sync_engine
