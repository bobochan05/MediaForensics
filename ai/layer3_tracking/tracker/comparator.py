from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ComparisonResult:
    previous_total: int
    current_total: int
    new_sources: list[str]
    existing_sources: list[str]

    @property
    def new_sources_count(self) -> int:
        return len(self.new_sources)


def compare_sources(existing_urls: set[str], current_urls: list[str]) -> ComparisonResult:
    current_set = set(current_urls)
    new_sources = sorted(current_set - existing_urls)
    existing_sources = sorted(current_set & existing_urls)
    return ComparisonResult(
        previous_total=len(existing_urls),
        current_total=len(current_set),
        new_sources=new_sources,
        existing_sources=existing_sources,
    )
