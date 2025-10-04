from bdlaw.parser.list_segmenter import split_semicolon_list


def test_split_semicolon_list_handles_simple_case():
    text = "возмещением убытков; взысканием неустойки; иными способами."
    segments = split_semicolon_list(text)
    assert [segment.text for segment in segments] == [
        "возмещением убытков",
        "взысканием неустойки",
        "иными способами.",
    ]


def test_split_semicolon_list_ignores_semicolons_in_parentheses():
    text = "исполнением; применением (см.; пояснения); иными"
    segments = split_semicolon_list(text)
    assert len(segments) == 3
    assert segments[1].text.startswith("применением")
