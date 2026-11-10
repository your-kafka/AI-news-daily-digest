import warnings
from typing import Any, Literal

from alembic import context
from alembic.autogenerate.api import AutogenContext
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning
from sqlalchemy.pool import NullPool

from ai_news.db.models import Base
from ai_news.db.session import database_url
from ai_news.log_config import configure_logging

warnings.filterwarnings("ignore", message="Did not recognize type 'geometry'", category=SAWarning)

target_metadata = Base.metadata


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    return type_ != "table" or name in target_metadata.tables


def render_item(type_: str, item: Any, autogen_context: AutogenContext) -> str | Literal[False]:
    if type_ == "type" and isinstance(item, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({item.dim})"
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        include_name=include_name,
        render_item=render_item,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
            render_item=render_item,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


configure_logging()
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
