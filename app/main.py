"""Entry point for the live interpreter.

Run with: python -m app.main
(the backend must already be running: uvicorn server.main:app)
"""
from __future__ import annotations

from .live_interpreter import LiveInterpreterApp


def main() -> None:
    LiveInterpreterApp().run()


if __name__ == "__main__":
    main()
