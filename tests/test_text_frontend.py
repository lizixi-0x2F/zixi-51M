"""Regression tests for the Chinese text frontend (CPU only, no model weights)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zixi.text import chinese  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "我喜欢 AI",
        "使用GPU计算",
        "Hello你好",
        "iPhone 15 很好",
        "WiFi连接",
        "café咖啡",
        "这是 test 句子",
    ],
)
def test_mixed_ascii_letters_do_not_crash(text):
    """Regression: ASCII letters used to trip `assert len(word2ph) == len(text)`."""
    norm = chinese.text_normalize(text)
    phones, tones, word2ph = chinese.g2p(norm)
    assert len(phones) == len(tones) > 0
    assert sum(word2ph) == len(phones)  # word2ph already counts the padding "_"


def test_pure_chinese_phones_unchanged():
    phones, tones, word2ph = chinese.g2p("你好世界")
    assert phones == ["_", "n", "i", "h", "ao", "sh", "ir", "j", "ie", "_"]
    assert tones == [0, 2, 2, 3, 3, 4, 4, 4, 4, 0]
    assert word2ph == [1, 2, 2, 2, 2, 1]


def test_text_normalize_numbers_and_punct():
    assert chinese.text_normalize("测试123") == "测试一百二十三"
    assert chinese.text_normalize("~~~") == "---"
