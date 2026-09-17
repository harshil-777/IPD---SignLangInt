"""Client abstraction for obtaining gesture predictions and generated
incident-report text from the backend.

Kept behind an interface (dependency inversion) so the live app depends
on "something that can predict and generate reports," not on HTTP or
`requests` specifically — a future in-process or gRPC client can drop in
without touching ``LiveInterpreterApp``.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

import requests

from .config import SERVER


class PredictionClient(ABC):
    @abstractmethod
    def request_prediction_async(self, sequence: list[list[float]]) -> None: ...

    @abstractmethod
    def consume_prediction(self) -> tuple[str, float, bool]: ...

    @abstractmethod
    def generate_report(self, compiled_inputs: str, callback: Callable[[str], None]) -> None: ...


class HttpPredictionClient(PredictionClient):
    """Fire-and-forget HTTP client. Predictions run on a background
    thread so the video loop never blocks on network I/O.
    """

    def __init__(self, base_url: str = SERVER.base_url, predict_timeout: float = 2.0, report_timeout: float = 30.0):
        self._base_url = base_url
        self._predict_timeout = predict_timeout
        self._report_timeout = report_timeout
        self._session = requests.Session()

        self._lock = threading.Lock()
        self._in_flight = False
        self._result_ready = False
        self._latest_result: tuple[str, float] = ("", 0.0)

    def request_prediction_async(self, sequence: list[list[float]]) -> None:
        with self._lock:
            if self._in_flight:
                return
            self._in_flight = True

        seq_copy = [row[:] for row in sequence]
        threading.Thread(target=self._predict, args=(seq_copy,), daemon=True).start()

    def _predict(self, sequence: list[list[float]]) -> None:
        try:
            resp = self._session.post(
                f"{self._base_url}/predict",
                json={"sequence": sequence},
                timeout=self._predict_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                with self._lock:
                    self._latest_result = (data.get("prediction", ""), data.get("confidence", 0.0))
                    self._result_ready = True
            else:
                print(f"[WARN] Backend returned status {resp.status_code}")
        except requests.RequestException as e:
            print(f"[WARN] Backend prediction request failed: {e}")
        finally:
            with self._lock:
                self._in_flight = False

    def consume_prediction(self) -> tuple[str, float, bool]:
        """Returns (prediction, confidence, is_new); is_new is True at
        most once per completed prediction.
        """
        with self._lock:
            if self._result_ready:
                self._result_ready = False
                return self._latest_result[0], self._latest_result[1], True
            return "", 0.0, False

    def generate_report(self, compiled_inputs: str, callback: Callable[[str], None]) -> None:
        def _fetch() -> None:
            try:
                resp = self._session.post(
                    f"{self._base_url}/generate_report",
                    json={"compiled_inputs": compiled_inputs},
                    timeout=self._report_timeout,
                )
                if resp.status_code == 200:
                    callback(resp.json().get("generated_sentence", ""))
                else:
                    callback(f"Error: Backend returned status {resp.status_code}")
            except Exception as e:
                callback(f"Error generating Incident Report: {e}")

        threading.Thread(target=_fetch, daemon=True).start()
