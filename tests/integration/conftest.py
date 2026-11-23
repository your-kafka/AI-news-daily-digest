import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from ai_news import config
from ai_news.config import PROJECT_ROOT
from ai_news.db import session as db_session

TEST_DATABASE = "lodestar_test"

pytestmark = pytest.mark.integration


def clear_cached_settings() -> None:
    config.database_settings.cache_clear()
    db_session.get_engine.cache_clear()
    db_session.get_session_factory.cache_clear()


def alembic_config() -> Config:
    return Config(toml_file=PROJECT_ROOT / "pyproject.toml")


@pytest.fixture(scope="session")
def database() -> Iterator[Engine]:
    """A fresh lodestar_test database, migrated to head. The real database is never touched."""
    previous = os.environ.get("POSTGRES_DB")
    clear_cached_settings()
    admin_url = db_session.database_url().set(database="postgres")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
            connection.execute(text(f"CREATE DATABASE {TEST_DATABASE}"))
    except OperationalError as error:
        admin.dispose()
        pytest.skip(f"PostgreSQL is not reachable; start it with `docker compose up -d` ({error.orig})")

    os.environ["POSTGRES_DB"] = TEST_DATABASE
    clear_cached_settings()
    assert db_session.database_url().database == TEST_DATABASE
    command.upgrade(alembic_config(), "head")
    yield db_session.get_engine()

    db_session.get_engine().dispose()
    if previous is None:
        os.environ.pop("POSTGRES_DB", None)
    else:
        os.environ["POSTGRES_DB"] = previous
    clear_cached_settings()
    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture
def db(database: Engine) -> Iterator[Engine]:
    """Empties every table before each test."""
    with database.begin() as connection:
        connection.execute(text("TRUNCATE documents, ingest_runs, digests RESTART IDENTITY CASCADE"))
    yield database
