"""Celery app: background jobs on Redis, split into queues so slow scrapers never starve indexing.

    celery -A ai_news.worker.celery_app worker -Q ingest -P threads -c 4 -n ingest@%h  # scrapers: wait on the network
    celery -A ai_news.worker.celery_app worker -Q index -P solo -n index@%h            # chunk, embed, digest: CPU
    celery -A ai_news.worker.celery_app beat                                         # the schedule below

The index worker runs one task at a time in one process (`solo`): embedding runs never race each other, and the
models load once without forking a process that already holds them.
"""

from typing import Any

from celery import Celery
from celery.schedules import crontab

from ai_news.config import WorkerSettings, worker_settings

INGEST_QUEUE = "ingest"
INDEX_QUEUE = "index"


def beat_schedule(settings: WorkerSettings) -> dict[str, dict[str, Any]]:
    return {
        "refresh": {
            "task": "ingest.refresh",
            "schedule": crontab(minute=0, hour=f"*/{settings.ingest_every_hours}"),
            "kwargs": {"hours": settings.ingest_window_hours},
        },
        "daily-digests": {
            "task": "digest.build_all",
            "schedule": crontab(minute=0, hour=settings.digest_hour),
        },
        "subscriber-digests": {
            "task": "digest.send_subscribers",
            "schedule": crontab(minute=0, hour=settings.digest_hour),
        },
    }


def create_celery(settings: WorkerSettings) -> Celery:
    app = Celery("lodestar", include=["ai_news.worker.tasks"])
    app.conf.update(
        broker_url=settings.redis_url,
        result_backend=settings.redis_url,
        result_expires=24 * 3600,
        task_acks_late=True,  # a task lost with its worker goes back on the queue
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # long tasks: don't let one worker hoard the queue
        task_track_started=True,
        task_routes={
            "ingest.source": {"queue": INGEST_QUEUE},
            "email.*": {"queue": INGEST_QUEUE},  # network-bound, like the scrapers
            "*": {"queue": INDEX_QUEUE},
        },
        task_default_queue=INDEX_QUEUE,
        timezone=settings.schedule_timezone,
        enable_utc=True,
        beat_schedule=beat_schedule(settings),
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
    )
    return app


app = create_celery(worker_settings())
