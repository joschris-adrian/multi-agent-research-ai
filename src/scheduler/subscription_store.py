import json
import os
from datetime import datetime
from typing import Optional

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "subscriptions.json")


def _load() -> list:
    if not os.path.exists(STORE_PATH):
        return []
    with open(STORE_PATH) as f:
        return json.load(f)


def _save(subscriptions: list):
    with open(STORE_PATH, "w") as f:
        json.dump(subscriptions, f, indent=2)


def add_subscription(topic: str, frequency: str, delivery_method: str, delivery_target: str) -> dict:
    subscriptions = _load()
    sub = {
        "id": f"{topic[:20].replace(' ', '_')}_{int(datetime.utcnow().timestamp() * 1000)}",
        "topic": topic,
        "frequency": frequency,
        "delivery_method": delivery_method,
        "delivery_target": delivery_target,
        "last_run": None,
        "paused": False,
        "created_at": datetime.utcnow().isoformat(),
    }
    subscriptions.append(sub)
    _save(subscriptions)
    return sub


def list_subscriptions() -> list:
    return _load()


def get_subscription(sub_id: str) -> Optional[dict]:
    return next((s for s in _load() if s["id"] == sub_id), None)


def pause_subscription(sub_id: str) -> bool:
    subscriptions = _load()
    for sub in subscriptions:
        if sub["id"] == sub_id:
            sub["paused"] = True
            _save(subscriptions)
            return True
    return False


def resume_subscription(sub_id: str) -> bool:
    subscriptions = _load()
    for sub in subscriptions:
        if sub["id"] == sub_id:
            sub["paused"] = False
            _save(subscriptions)
            return True
    return False


def delete_subscription(sub_id: str) -> bool:
    subscriptions = _load()
    updated = [s for s in subscriptions if s["id"] != sub_id]
    if len(updated) == len(subscriptions):
        return False
    _save(updated)
    return True


def update_last_run(sub_id: str):
    subscriptions = _load()
    for sub in subscriptions:
        if sub["id"] == sub_id:
            sub["last_run"] = datetime.utcnow().isoformat()
            _save(subscriptions)
            return