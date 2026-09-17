"""State machine for the step-by-step FIR (First Information Report)
sign-language questionnaire.
"""
from __future__ import annotations

FIR_STEPS = [
    "Step 1: Your Name? (Spell it)",
    "Step 2: What Happened? (Theft, Attack, Kidnap)",
    "Step 3: When did it happen? (HH:MM)",
    "Step 4: Where did it happen? (Location)",
    "Step 5: Any vehicles involved? (Car, Bike, Truck)",
    "Step 6: Describe the actions (Push, Pull, Drink, Drive)",
    "Step 7: Stolen items or injuries?",
    "Step 8: Describe the accused.",
    "Step 9: Any evidence or witnesses?",
    "Step 10: Any other information?",
]

_TIME_STEP_INDEX = 2


def _format_time(words: list[str]) -> str:
    raw = "".join(words).replace(" ", "").replace(":", "")
    try:
        if len(raw) >= 2:
            hh = int(raw[:2])
            mm = raw[2:] if len(raw) > 2 else "00"
        else:
            hh = int(raw)
            mm = "00"

        period = "AM" if hh < 12 else "PM"
        display_hh = hh if hh <= 12 else hh - 12
        display_hh = 12 if display_hh == 0 else display_hh
        return f"{display_hh}:{mm.ljust(2, '0')} {period}"
    except ValueError:
        return " ".join(words)


class FIRWorkflow:
    """Tracks answers across the fixed FIR question sequence."""

    def __init__(self):
        self.current_step = 0
        self._answers: dict[int, list[str]] = {i: [] for i in range(len(FIR_STEPS))}

    @property
    def current_question(self) -> str:
        return FIR_STEPS[self.current_step]

    def save_current(self, words: list[str]) -> None:
        self._answers[self.current_step] = list(words)

    def answer_for_current_step(self) -> list[str]:
        return list(self._answers[self.current_step])

    def next_step(self, words: list[str]) -> list[str]:
        self.save_current(words)
        if self.current_step < len(FIR_STEPS) - 1:
            self.current_step += 1
        return self.answer_for_current_step()

    def previous_step(self, words: list[str]) -> list[str]:
        self.save_current(words)
        if self.current_step > 0:
            self.current_step -= 1
        return self.answer_for_current_step()

    def compile_report_input(self, words: list[str]) -> str:
        self.save_current(words)
        lines = []
        for i, question in enumerate(FIR_STEPS):
            answer_words = self._answers[i]
            if not answer_words:
                signs = "N/A"
            elif i == _TIME_STEP_INDEX:
                signs = _format_time(answer_words)
            else:
                signs = " ".join(answer_words)
            lines.append(f"{question}: {signs}")
        return "\n".join(lines) + "\n"
