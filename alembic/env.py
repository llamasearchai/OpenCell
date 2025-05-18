import os
import sys
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# This line allows the script to find your project's modules
# Adjust the path as necessary if your project structure is different
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.workflow.database.db import DATABASE_URL  # Import the configured URL
from src.workflow.database.models import Base  # noqa

# We use the synchronous one for migrations

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# For 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None
target_metadata = Base.metadata

# Define the database URL, prioritizing environment variable, then a default.
# Ensure the URL is synchronous for Alembic.
# app_db_url = os.environ.get("DATABASE_URL", "sqlite:///./opencell.db") # Original problematic line
# if app_db_url.startswith("sqlite+aiosqlite://"):
# app_db_url = app_db_url.replace("sqlite+aiosqlite://", "sqlite:///")

# Use the imported DATABASE_URL which should already be synchronous for Alembic
configured_db_url = DATABASE_URL
if configured_db_url.startswith("sqlite+aiosqlite://"):
    configured_db_url = configured_db_url.replace("sqlite+aiosqlite://", "sqlite://")

# other values from the config, defined by the needs of env.py,
# can be acquired: # my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # url = config.get_main_option("sqlalchemy.url") # Original line
    url = configured_db_url  # Use the processed URL
    context.configure(
        url=url,  # Use the processed URL
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schema an option for multi-schema support
        # version_table_schema=target_metadata.schema,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = create_engine(configured_db_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Compare types when generating migrations
            render_as_batch=True,  # Recommended for SQLite
            # Include schema an option for multi-schema support
            # version_table_schema=target_metadata.schema,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
