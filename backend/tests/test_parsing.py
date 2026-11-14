from parsing import StopMatcher, ThinkingSplitter


# ---------------------------------------------------------------- StopMatcher
def feed_all(matcher, chunks):
    out = ""
    for chunk in chunks:
        safe, match = matcher.feed(chunk)
        out += safe
        if match:
            return out, match
    return out, None


def test_stop_matcher_without_stops_passes_everything():
    matcher = StopMatcher([])
    assert matcher.feed("hello") == ("hello", None)
    assert matcher.flush() == ""


def test_stop_matcher_holds_partial_suffix():
    matcher = StopMatcher(["END"])
    assert matcher.feed("hello E") == ("hello ", None)
    assert matcher.feed("N") == ("", None)
    safe, match = matcher.feed("D rest")
    assert safe == ""
    assert match is not None
    assert match.sequence == "END"
    assert match.offset == 6


def test_stop_matcher_does_not_hold_false_prefix():
    matcher = StopMatcher(["END"])
    assert matcher.feed("hello Ex") == ("hello Ex", None)
    assert matcher.flush() == ""


def test_stop_matcher_earliest_sequence_wins_and_offset_is_global():
    matcher = StopMatcher(["STOP", "END"])
    safe, match = feed_all(matcher, ["abc ", "ST", "OP then END"])
    assert safe == "abc "
    assert match.sequence == "STOP"
    assert match.offset == 4


def test_stop_matcher_flush_releases_held_text():
    matcher = StopMatcher(["END"])
    assert matcher.feed("abc E") == ("abc ", None)
    assert matcher.flush() == "E"


def test_stop_matcher_single_character_stop():
    matcher = StopMatcher(["\n"])
    safe, match = matcher.feed("one\ntwo")
    assert safe == "one"
    assert match is not None and match.offset == 3


# ------------------------------------------------------------ ThinkingSplitter
def collect(splitter, chunks):
    """Feed several chunks and merge adjacent same-channel segments.

    ``feed`` returns only the segments produced by that call (which is what the
    streaming loop needs).  The tests care about the final channel contents, so
    concatenate neighbours on the same channel.
    """
    out = []
    for chunk in chunks:
        for channel, text in splitter.feed(chunk):
            if not text:
                continue
            if out and out[-1][0] == channel:
                out[-1] = (channel, out[-1][1] + text)
            else:
                out.append((channel, text))
    return out


def test_thinking_splitter_start_in_thinking_and_close():
    splitter = ThinkingSplitter("<think>", "</think>", starts_in_thinking=True)
    segments = collect(splitter, ["rea", "son</thi", "nk>ans", "wer"])
    assert segments == [("thinking", "reason"), ("answer", "answer")]
    assert splitter.closed is True
    assert splitter.flush() == []


def test_thinking_splitter_start_in_answer_then_marker():
    splitter = ThinkingSplitter("<think>", "</think>", starts_in_thinking=False)
    segments = collect(splitter, ["be", "fore<thi", "nk>in</think>", "after"])
    assert segments == [("answer", "before"), ("thinking", "in"), ("answer", "after")]


def test_thinking_splitter_holds_partial_marker_at_flush():
    splitter = ThinkingSplitter("<think>", "</think>", starts_in_thinking=True)
    assert splitter.feed("done</thi") == [("thinking", "done")]
    assert splitter.flush() == [("thinking", "</thi")]


def test_thinking_splitter_never_closes_means_no_answer():
    splitter = ThinkingSplitter("<think>", "</think>", starts_in_thinking=True)
    segments = collect(splitter, ["only thinking ", "here"])
    assert segments == [("thinking", "only thinking here")]
    assert splitter.closed is False


def test_thinking_splitter_without_markers_is_answer_only():
    splitter = ThinkingSplitter(None, None, starts_in_thinking=False)
    assert splitter.feed("plain text") == [("answer", "plain text")]


def test_thinking_splitter_marker_straddles_three_chunks():
    splitter = ThinkingSplitter("<think>", "</think>", starts_in_thinking=True)
    segments = collect(splitter, ["a</", "thi", "nk>b"])
    assert segments == [("thinking", "a"), ("answer", "b")]
