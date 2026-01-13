"""AVP-ViT inference app.

Suppress streamlit "missing ScriptRunContext" warnings when imported outside streamlit.
This happens when running pypatree or other tools that import the module.
"""

import logging
import os
import sys

# Suppress streamlit's chatty logging when not running as streamlit app
os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")

# Also suppress via logging module (streamlit uses both)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


# Suppress streamlit's direct stderr writes
class _StreamlitStderrFilter:
    """Wrap stderr to filter streamlit's ScriptRunContext warnings."""

    def __init__(self, original: object):
        self._original = original

    def write(self, msg: str) -> int:
        if "ScriptRunContext" in msg:
            return len(msg)  # pretend we wrote it
        return self._original.write(msg)  # type: ignore

    def flush(self) -> None:
        self._original.flush()  # type: ignore

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


# Only filter when not running as streamlit app
if "streamlit" not in sys.modules:
    sys.stderr = _StreamlitStderrFilter(sys.stderr)  # type: ignore
