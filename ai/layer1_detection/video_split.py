from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sklearn.model_selection import train_test_split

from ai.layer1_detection.data_loader import VideoRecord
from ai.shared.file_utils import save_json


def split_video_records(
    video_records: Sequence[VideoRecord],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    labels = [record.label for record in video_records]
    train_records, test_records = train_test_split(
        list(video_records),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    validate_no_video_overlap(train_records, test_records)
    return sorted(train_records, key=lambda record: record.video_id), sorted(test_records, key=lambda record: record.video_id)


def validate_no_video_overlap(
    train_records: Sequence[VideoRecord],
    test_records: Sequence[VideoRecord],
) -> None:
    train_ids = {record.video_id for record in train_records}
    test_ids = {record.video_id for record in test_records}
    overlap = train_ids.intersection(test_ids)

    if overlap:
        raise ValueError(f"Video leakage detected across splits: {sorted(overlap)[:5]}")


def save_split_manifest(
    output_path: str | Path,
    train_records: Sequence[VideoRecord],
    test_records: Sequence[VideoRecord],
) -> None:
    save_json(
        output_path,
        {
            "train": [_record_to_dict(record) for record in train_records],
            "test": [_record_to_dict(record) for record in test_records],
        },
    )


def _record_to_dict(record: VideoRecord) -> dict[str, str | int]:
    return {
        "video_id": record.video_id,
        "video_path": str(record.video_path),
        "label": record.label,
        "source": record.source,
    }
