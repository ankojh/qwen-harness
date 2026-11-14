"""Text-level parsing of the decoded token stream.

There are two independent layers here, both needed because tokens arrive in
pieces and neither stop strings nor think markers respect token boundaries.

1. ``StopMatcher`` sits on the raw decoded text.  It holds back the longest
   suffix that could still become a stop string, so a stop string is never
   streamed out and then retracted.  When it matches, it reports the sequence
   and the character offset and drops everything from that point on.

2. ``ThinkingSplitter`` splits the same text into ``thinking`` and ``answer``
   channels.  It also holds back a possible partial marker (for example a lone
   ``"<"`` that might grow into ``"</think>"``).  Markers themselves are
   removed from both channels; the raw text kept by the caller still has them.

Both classes are pure string state machines: no MLX, no I/O, easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StopMatch:
    sequence: str
    offset: int  # character offset in the text fed so far


class StopMatcher:
    def __init__(self, stop_sequences: list[str]):
        self.stops = [s for s in stop_sequences if s]
        self._pending = ""
        self._consumed = 0  # characters already returned from feed()
        self._match: StopMatch | None = None

    @property
    def match(self) -> StopMatch | None:
        return self._match

    def feed(self, text: str) -> tuple[str, StopMatch | None]:
        """Return (safe text, match).  Text after a match is discarded."""
        if self._match is not None:
            return "", self._match

        if not self.stops:
            self._consumed += len(text)
            return text, None

        self._pending += text

        best_index = -1
        best_sequence = ""
        for sequence in self.stops:
            index = self._pending.find(sequence)
            if index != -1 and (
                best_index == -1
                or index < best_index
                or (index == best_index and len(sequence) > len(best_sequence))
            ):
                best_index, best_sequence = index, sequence

        if best_index != -1:
            safe = self._pending[:best_index]
            self._consumed += len(safe)
            self._pending = ""
            self._match = StopMatch(best_sequence, self._consumed)
            return safe, self._match

        # Hold back the longest suffix that is a proper prefix of a stop string.
        hold = 0
        for sequence in self.stops:
            for size in range(min(len(sequence) - 1, len(self._pending)), hold, -1):
                if self._pending.endswith(sequence[:size]):
                    hold = size
                    break

        if hold:
            safe = self._pending[:-hold]
            self._pending = self._pending[-hold:]
        else:
            safe = self._pending
            self._pending = ""

        self._consumed += len(safe)
        return safe, None

    def flush(self) -> str:
        """Release held text at the end of a run (no stop matched)."""
        if self._match is not None:
            # The match and everything after it stay out of the channels.
            self._pending = ""
            return ""
        out = self._pending
        self._pending = ""
        self._consumed += len(out)
        return out


class ThinkingSplitter:
    def __init__(
        self,
        think_start: str | None,
        think_end: str | None,
        starts_in_thinking: bool = False,
    ):
        self.start = think_start or ""
        self.end = think_end or ""
        self.in_thinking = bool(starts_in_thinking) and bool(self.start and self.end)
        self.closed = not self.in_thinking
        self._pending = ""

    @property
    def channel(self) -> str:
        return "thinking" if self.in_thinking else "answer"

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Return ordered (channel, text) segments with markers removed."""
        if not self.start or not self.end:
            if not text:
                return []
            return [("answer", text)]

        self._pending += text
        segments: list[tuple[str, str]] = []

        while True:
            marker = self.end if self.in_thinking else self.start
            index = self._pending.find(marker)

            if index != -1:
                if index:
                    segments.append((self.channel, self._pending[:index]))
                self._pending = self._pending[index + len(marker) :]
                self.in_thinking = not self.in_thinking
                self.closed = not self.in_thinking
                continue

            # Hold back a suffix that might still become the active marker.
            hold = 0
            for size in range(min(len(marker) - 1, len(self._pending)), 0, -1):
                if self._pending.endswith(marker[:size]):
                    hold = size
                    break

            if hold:
                if len(self._pending) > hold:
                    segments.append((self.channel, self._pending[:-hold]))
                self._pending = self._pending[-hold:]
            elif self._pending:
                segments.append((self.channel, self._pending))
                self._pending = ""
            break

        return segments

    def flush(self) -> list[tuple[str, str]]:
        if not self._pending:
            return []
        out = [(self.channel, self._pending)]
        self._pending = ""
        return out
