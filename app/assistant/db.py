from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from app.assistant.config import settings

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
except Exception:  # pragma: no cover - dependency may be absent in old envs
    create_engine = None
    Session = object  # type: ignore
    sessionmaker = None

    class DeclarativeBase:  # type: ignore
        pass


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None
_init_error: Optional[str] = None


def is_database_configured() -> bool:
    return bool(settings.database_url)


def is_database_available() -> bool:
    return bool(settings.database_url and create_engine is not None)


def get_database_init_error() -> Optional[str]:
    return _init_error


def get_engine():
    global _engine, _init_error
    if not is_database_configured():
        return None
    if create_engine is None:
        _init_error = "SQLAlchemy is not installed. Install requirements.txt first."
        return None
    if _engine is None:
        try:
            _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
        except Exception as exc:
            _init_error = str(exc)
            return None
    return _engine


def init_db() -> bool:
    global _init_error
    engine = get_engine()
    if engine is None:
        return False
    try:
        from sqlalchemy import text

        from app.assistant import models  # noqa: F401

        # pgvector chỉ cần cho bảng opinion_chunks (tìm kiếm ngữ nghĩa); các
        # bảng quan hệ không dùng kiểu vector. Khi Postgres cục bộ chưa cài
        # pgvector, vẫn bật phần lưu trữ/thống kê, và PgVectorOpinionStore sẽ
        # tự tắt riêng phần vector.
        try:
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:
            print(f"[V2 DB] pgvector chưa sẵn sàng, tắt tìm kiếm ngữ nghĩa: {str(exc).splitlines()[0]}")
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as exc:
        _init_error = str(exc)
        return False


@contextmanager
def session_scope() -> Iterator[Session]:
    global _SessionLocal
    engine = get_engine()
    if engine is None or sessionmaker is None:
        raise RuntimeError(get_database_init_error() or "DATABASE_URL is not configured")
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
