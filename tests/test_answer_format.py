import pytest

from bdlaw.gpt.answerer import _validate_answer_format


def test_validate_answer_accepts_correct_format():
    answer = "[ГК РФ ст. 12] — Текст ответа. Источник: https://example.com"
    _validate_answer_format(answer)


def test_validate_answer_rejects_wrong_format():
    with pytest.raises(ValueError):
        _validate_answer_format("неправильный ответ")
