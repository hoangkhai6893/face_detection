# src/services — Background Services

## Modules

| File | Role |
|---|---|
| `notification_worker.py` | Daemon thread — queues and sends notifications async |
| `alert_manager.py` | Alert rules: define conditions → trigger notifications |
| `event_logger.py` | Structured event logging (recognition hits, unknowns, errors) |
| `device_dispatcher.py` | Route output to multiple devices (display, file, network) |

## Pattern

All services follow the same daemon thread pattern — see `notification_worker.py` as reference:
- Started as `daemon=True` — dies when main process exits
- Communicate via `queue.Queue` — no shared mutable state
- Caller puts events on queue; worker drains and dispatches
