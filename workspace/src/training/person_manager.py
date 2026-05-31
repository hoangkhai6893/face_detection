#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import config
from core.encoder import rebuild_encodings

logger = logging.getLogger(__name__)

_INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]|^\.|^\.\.')
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PersonInfo:
    name: str
    image_count: int
    augmented_count: int
    folder_path: Path
    created_date: str


class PersonManager:
    """Create, list, delete, and reset person folders in the dataset."""

    def __init__(self, dataset_path: str = config.DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_persons(self) -> List[PersonInfo]:
        persons = []
        for entry in sorted(self.dataset_path.iterdir()):
            if not entry.is_dir():
                continue
            images = [
                f for f in entry.iterdir()
                if f.suffix.lower() in _IMAGE_EXTS
            ]
            aug = [f for f in images if "_aug_" in f.name]
            orig = [f for f in images if "_aug_" not in f.name]
            created = time.strftime(
                "%Y-%m-%d", time.localtime(entry.stat().st_ctime)
            )
            persons.append(PersonInfo(
                name=entry.name,
                image_count=len(orig),
                augmented_count=len(aug),
                folder_path=entry,
                created_date=created,
            ))
        return persons

    def get_image_count(self, name: str) -> int:
        folder = self.dataset_path / name
        if not folder.is_dir():
            return 0
        return sum(
            1 for f in folder.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and "_aug_" not in f.name
        )

    def get_augmented_count(self, name: str) -> int:
        folder = self.dataset_path / name
        if not folder.is_dir():
            return 0
        return sum(
            1 for f in folder.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and "_aug_" in f.name
        )

    def exists(self, name: str) -> bool:
        return (self.dataset_path / name).is_dir()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create_person(self, name: str) -> Path:
        """Create a new person folder. Raises ValueError for invalid names."""
        if _INVALID_NAME_RE.search(name) or not name.strip():
            raise ValueError(f"Invalid person name: {name!r}")
        path = self.dataset_path / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Created person folder: %s", path)
        return path

    def delete_person(self, name: str, rebuild: bool = True) -> bool:
        """Delete a person folder and optionally rebuild encodings."""
        path = self.dataset_path / name
        if not path.is_dir():
            logger.warning("Person not found: %s", name)
            return False
        shutil.rmtree(path)
        logger.info("Deleted person: %s", name)
        if rebuild:
            self._rebuild()
        return True

    def reset_person(self, name: str, rebuild: bool = True) -> int:
        """Delete all images for a person (keep folder). Returns image count deleted."""
        path = self.dataset_path / name
        if not path.is_dir():
            logger.warning("Person not found: %s", name)
            return 0
        images = [
            f for f in path.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        ]
        for f in images:
            f.unlink()
        logger.info("Reset person %s: deleted %d images", name, len(images))
        if rebuild:
            self._rebuild()
        return len(images)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        encodings_path = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
        logger.info("Rebuilding encodings → %s", encodings_path)
        total_enc, total_persons = rebuild_encodings(
            dataset_path=str(self.dataset_path),
            model_path=config.MODEL_PATH,
            encodings_path=encodings_path,
            logger=logger,
        )
        logger.info(
            "Encodings rebuilt — %d encodings for %d persons",
            total_enc, total_persons,
        )
