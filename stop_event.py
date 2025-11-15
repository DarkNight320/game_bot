"""A tiny stop-event helper used by the bot modules.

This module exposes a simple module-level boolean `STOP_EVENT` so files that
do `from stop_event import STOP_EVENT` can import it without error. Note:

- Current code often assigns to `STOP_EVENT` (e.g. `STOP_EVENT = True`) in
  other modules. That pattern rebinds the name in the assigning module and
  does not change this module's variable. A safer cross-module pattern is to
  either import the module (`import stop_event`) and set
  `stop_event.STOP_EVENT = True`, or use a mutable object (e.g. threading.Event).

This file keeps the minimal `STOP_EVENT` boolean for compatibility and also
provides helper functions you can use to migrate safely.
"""

STOP_EVENT = False

def set_stop():
    """Set the shared stop flag.

    Prefer calling this as `import stop_event; stop_event.set_stop()` so the
    module-level variable is updated in-place.
    """
    global STOP_EVENT
    STOP_EVENT = True

def clear_stop():
    """Clear the shared stop flag."""
    global STOP_EVENT
    STOP_EVENT = False

def is_stopped():
    """Return the current stop flag as a boolean."""
    return bool(STOP_EVENT)
