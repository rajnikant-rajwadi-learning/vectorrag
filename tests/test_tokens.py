from vectorrag.tokens import count_tokens, truncate_to_tokens


def test_count_tokens_basic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


def test_truncate_shorter_than_limit_unchanged():
    text = "a short string"
    assert truncate_to_tokens(text, 100) == text


def test_truncate_enforces_limit():
    text = "word " * 500
    out = truncate_to_tokens(text, 10)
    assert count_tokens(out) <= 10
