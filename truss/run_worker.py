"""Entry-point for running a Temporal worker that executes Truss workflows.

Usage::

    # Ensure DATABASE_URL and TEMPORAL_URL are exported, then run
    $ python -m truss.run_worker

Environment variables
DATABASE_URL
    SQLAlchemy database URL.  Defaults to ``sqlite:///truss.db`` for local dev.
TEMPORAL_URL
    Target Temporal frontend host:port. Defaults to ``localhost:7233``.
TEMPORAL_TASK_QUEUE
    Task queue name worker should poll. Defaults to ``truss-agent-queue``.
"""
from __future__ import annotations

import asyncio
import os
from typing import Sequence, Callable, Any

from temporalio.client import Client
from temporalio.worker import Worker
from truss.activities.storage_activities import StorageActivities
from truss.activities.llm_activities import LLMActivities
from truss.activities.tool_activities import execute_tool_activity, get_tools

from truss.core.storage import PostgresStorage
from truss.workflows.agent_workflow import TemporalAgentExecutionWorkflow




def _collect_storage_activity_fns(svc: StorageActivities) -> Sequence[Callable[..., Any]]:
    """Return bound *activity* functions declared on *StorageActivities* instance."""

    return [
        svc.create_run,
        svc.create_run_step,
        svc.get_run_memory,
        svc.load_agent_config,
        svc.finalize_run,
    ]


async def main() -> None:  # noqa: D401 – imperative mood
    """Bootstrap and run the Temporal worker indefinitely."""

    db_url = os.getenv("DATABASE_URL", "sqlite:///truss.db")
    temporal_address = os.getenv("TEMPORAL_URL", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "truss-agent-queue")

    
    storage = PostgresStorage.from_database_url(db_url)
    storage_activities = StorageActivities(storage)

    llm_activities = LLMActivities(storage)

   
    print(f"[worker] Connecting to Temporal at {temporal_address}…", flush=True)
    client = await Client.connect(temporal_address)

    
    activities = [
        *_collect_storage_activity_fns(storage_activities),
        llm_activities.llm_activity, 
        execute_tool_activity,
        get_tools,
    ]

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[TemporalAgentExecutionWorkflow],
        activities=activities,
    )

    print(
        f"[worker] Starting worker polling queue '{task_queue}'."
        " Press Ctrl+C to exit.",
        flush=True,
    )

    await worker.run()


if __name__ == "__main__":  # pragma: no cover – script entry
    asyncio.run(main()) 
