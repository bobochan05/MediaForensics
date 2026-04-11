from ai.layer3_tracking.tracker.comparator import compare_sources


def test_compare_sources_identifies_new_and_existing_urls():
    existing = {"https://example.com/a", "https://example.com/b"}
    current = [
        "https://example.com/a",
        "https://example.com/c",
        "https://example.com/c",
    ]

    result = compare_sources(existing, current)

    assert result.previous_total == 2
    assert result.current_total == 2
    assert result.new_sources == ["https://example.com/c"]
    assert result.existing_sources == ["https://example.com/a"]
    assert result.new_sources_count == 1


def test_compare_sources_handles_first_run_without_previous_sources():
    result = compare_sources(set(), ["https://example.com/a", "https://example.com/b"])

    assert result.previous_total == 0
    assert result.current_total == 2
    assert result.new_sources == ["https://example.com/a", "https://example.com/b"]
    assert result.existing_sources == []
