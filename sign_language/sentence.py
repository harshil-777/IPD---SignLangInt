"""Accumulates recognized signs into words and a running sentence."""
from __future__ import annotations


class SentenceBuilder:
    """Merges consecutive single-letter signs (finger-spelling) into one
    word, and appends multi-letter signs (whole-word gestures) as their
    own word.
    """

    _MAX_SPELLED_WORD_LEN = 10

    def __init__(self):
        self.words: list[str] = []
        self._last_added = ""

    def add(self, label: str) -> None:
        if not label or label == self._last_added:
            return

        is_letter = len(label) == 1
        can_extend = (
            is_letter
            and self.words
            and len(self.words[-1]) <= self._MAX_SPELLED_WORD_LEN
            and len(self._last_added) == 1
        )
        if can_extend:
            self.words[-1] += label
        else:
            self.words.append(label)

        self._last_added = label

    def delete_last(self) -> None:
        if not self.words:
            return
        if len(self._last_added) == 1 and len(self.words[-1]) > 1:
            self.words[-1] = self.words[-1][:-1]
        else:
            self.words.pop()
        self._last_added = "DELETED_LOCK"

    def clear(self) -> None:
        self.words = []
        self._last_added = ""

    def set_words(self, words: list[str]) -> None:
        self.words = list(words)
        self._last_added = ""

    def reset_repeat_guard(self) -> None:
        """Allow the next recognized sign to be appended even if it
        repeats the last one added (used after a hand-tracking gap).
        """
        self._last_added = ""

    def text(self, last_n: int | None = None) -> str:
        words = self.words[-last_n:] if last_n else self.words
        return " ".join(words)
