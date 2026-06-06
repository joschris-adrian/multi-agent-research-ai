import os
from datetime import datetime, timedelta
from src.scheduler.subscription_store import (
    list_subscriptions, update_last_run
)
from src.scheduler.delivery import deliver
from src.workflow.agent_pipeline import MultiAgentResearchSystem

FREQUENCY_DELTAS = {
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
}


def _is_due(subscription: dict) -> bool:
    if subscription.get("paused"):
        return False
    last_run = subscription.get("last_run")
    if not last_run:
        return True
    frequency = subscription.get("frequency", "weekly")
    delta = FREQUENCY_DELTAS.get(frequency, timedelta(weeks=1))
    last_run_dt = datetime.fromisoformat(last_run)
    return datetime.utcnow() >= last_run_dt + delta


def _submitted_after(subscription: dict) -> str:
    last_run = subscription.get("last_run")
    if not last_run:
        return ""
    return last_run[:10]


def run_due_subscriptions():
    """Check all subscriptions and run any that are due."""
    subscriptions = list_subscriptions()
    if not subscriptions:
        print("[scheduler] no subscriptions found")
        return

    system = MultiAgentResearchSystem()

    for sub in subscriptions:
        if not _is_due(sub):
            print(f"[scheduler] skipping {sub['topic']} — not due yet")
            continue

        print(f"[scheduler] running pipeline for topic: {sub['topic']}")
        try:
            result = system.run(sub["topic"])
            update_last_run(sub["id"])
            deliver(
                report=result["report"],
                topic=sub["topic"],
                delivery_method=sub["delivery_method"],
                delivery_target=sub["delivery_target"],
            )
            print(f"[scheduler] completed and delivered: {sub['topic']}")
        except Exception as e:
            print(f"[scheduler] pipeline failed for {sub['topic']}: {e}")