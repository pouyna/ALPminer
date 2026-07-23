import pytest

from alpminer import utils


class _Resp:
    def __init__(self, headers):
        self.headers = headers


class _HTTPish(Exception):
    """Stand-in for requests.HTTPError carrying an optional response."""
    def __init__(self, headers=None):
        super().__init__("boom")
        self.response = _Resp(headers) if headers is not None else None


def test_retry_after_header_is_honored(monkeypatch):
    slept = []
    monkeypatch.setattr(utils.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(utils.random, "uniform", lambda a, b: 0.0)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPish({"Retry-After": "7"})
        return "ok"

    assert utils.with_retries(fn, attempts=4, base_delay=2.0,
                              retry_on=(_HTTPish,)) == "ok"
    assert slept == [7.0, 7.0]        # server's Retry-After, not 2s/4s backoff


def test_retry_after_is_capped_at_max_delay(monkeypatch):
    slept = []
    monkeypatch.setattr(utils.time, "sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _HTTPish({"Retry-After": "9999"})
        return "ok"

    utils.with_retries(fn, attempts=3, base_delay=2.0, max_delay=60.0,
                       retry_on=(_HTTPish,))
    assert slept == [60.0]            # absurd Retry-After clamped to max_delay


def test_backoff_used_without_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr(utils.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(utils.random, "uniform", lambda a, b: 0.0)

    def fn():
        raise _HTTPish(None)          # no response/header at all

    with pytest.raises(utils.RetryError):
        utils.with_retries(fn, attempts=3, base_delay=2.0,
                           retry_on=(_HTTPish,))
    assert slept == [2.0, 4.0]        # exponential backoff between 3 attempts
