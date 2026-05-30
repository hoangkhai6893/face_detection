#!/usr/bin/env python3.11
"""
Smoke driver for the Family Face Recognition System.

Exercises core library modules without a camera or GUI.
Run from the workspace root:
    python3.11 .claude/skills/run-workspace/driver.py

Exit 0 = all checks passed.
"""
import logging
import os
import shutil
import sys
import tempfile

# ---- path setup -----------------------------------------------------------
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SRC = os.path.join(_WORKSPACE, "src")
sys.path.insert(0, _SRC)

import numpy as np
import cv2

logging.basicConfig(level=logging.WARNING)  # suppress YOLO/dlib noise during smoke
log = logging.getLogger("driver")

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        FAIL.append(name)
        print(f"  FAIL  {name}: {exc}")


# ---------------------------------------------------------------------------
# 1. Import check — all modules must be importable
# ---------------------------------------------------------------------------

def _import_check():
    import config
    from core.frame_extractor import FrameQualityChecker, FrameDiversityFilter
    from training_manager import PersonManager

check("imports", _import_check)


# ---------------------------------------------------------------------------
# 2. FrameQualityChecker — synthetic images
# ---------------------------------------------------------------------------

def _quality_checker():
    from core.frame_extractor import FrameQualityChecker
    checker = FrameQualityChecker()

    # Sharp, bright 200×200 face crop — should pass all gates
    sharp = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
    assert checker.is_sharp(sharp), "sharp image failed sharpness check"
    assert checker.is_bright(sharp), "bright image failed brightness check"
    assert checker.is_large_enough(sharp), "200px image failed size check"
    assert checker.passes_all(sharp), "sharp image failed passes_all"
    assert 0.0 <= checker.score(sharp) <= 1.0, "score out of range"

    # Solid black — should be rejected (too dark)
    black = np.zeros((200, 200, 3), dtype=np.uint8)
    assert not checker.is_bright(black), "black image passed brightness check"

check("FrameQualityChecker", _quality_checker)


# ---------------------------------------------------------------------------
# 3. FrameDiversityFilter — temporal deduplication
# ---------------------------------------------------------------------------

def _diversity_filter():
    from core.frame_extractor import FrameDiversityFilter
    fdf = FrameDiversityFilter(min_frame_gap=1)

    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)           # black
    frame_b = np.ones((100, 100, 3), dtype=np.uint8) * 255      # white

    assert fdf.is_diverse(frame_a, frame_num=0), "first frame should be diverse"
    fdf.accept(frame_a, frame_num=0)

    # Identical content should fail diversity check
    assert not fdf.is_diverse(frame_a.copy(), frame_num=1), "identical frame should not be diverse"

    # Very different content should pass
    assert fdf.is_diverse(frame_b, frame_num=2), "different frame should be diverse"

check("FrameDiversityFilter", _diversity_filter)


# ---------------------------------------------------------------------------
# 4. PersonManager CRUD — isolated temp dataset, no encoding rebuild
# ---------------------------------------------------------------------------

def _person_manager():
    from training_manager import PersonManager
    tmp = tempfile.mkdtemp(prefix="face_smoke_")
    try:
        pm = PersonManager(tmp)

        # Create
        path = pm.create_person("SmokePerson")
        assert os.path.isdir(str(path)), "person folder not created"
        assert pm.exists("SmokePerson")

        # List
        persons = pm.list_persons()
        assert any(p.name == "SmokePerson" for p in persons), "person not listed"

        # Image count (empty folder)
        assert pm.get_image_count("SmokePerson") == 0

        # Delete with rebuild=False so we don't touch the real encodings pkl
        ok = pm.delete_person("SmokePerson", rebuild=False)
        assert ok, "delete returned False"
        assert not pm.exists("SmokePerson"), "person still exists after delete"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

check("PersonManager CRUD", _person_manager)


# ---------------------------------------------------------------------------
# 5. augment_person — synthetic face image in temp dataset
# ---------------------------------------------------------------------------

def _augment_person():
    import augment_dataset as ad
    tmp = tempfile.mkdtemp(prefix="face_aug_")
    try:
        person_dir = os.path.join(tmp, "AugTest")
        os.makedirs(person_dir)

        # Synthetic 200×200 grayscale-like face (solid mid-grey)
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        cv2.imwrite(os.path.join(person_dir, "face.jpg"), img)

        aug_logger = logging.getLogger("aug_smoke")
        generated, skipped = ad.augment_person(
            person_name="AugTest",
            dataset_path=tmp,
            show_preview=False,
            logger=aug_logger,
        )
        assert generated > 0, f"expected augmented images, got {generated}"

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

check("augment_person", _augment_person)


# ---------------------------------------------------------------------------
# 6. core.encoder rebuild_encodings — empty temp dataset
# ---------------------------------------------------------------------------

def _rebuild_encodings_empty():
    from core.encoder import rebuild_encodings
    tmp_ds = tempfile.mkdtemp(prefix="face_enc_ds_")
    tmp_enc = tempfile.mkdtemp(prefix="face_enc_out_")
    enc_file = os.path.join(tmp_enc, "test_encodings.pkl")
    try:
        import config
        total_enc, total_persons = rebuild_encodings(
            dataset_path=tmp_ds,
            model_path=config.MODEL_PATH,
            encodings_path=enc_file,
            logger=logging.getLogger("enc_smoke"),
        )
        assert total_enc == 0, f"expected 0 encodings for empty dataset, got {total_enc}"
        assert total_persons == 0, f"expected 0 persons, got {total_persons}"
        assert os.path.exists(enc_file), "encodings file not created"
    finally:
        shutil.rmtree(tmp_ds, ignore_errors=True)
        shutil.rmtree(tmp_enc, ignore_errors=True)

check("rebuild_encodings (empty dataset)", _rebuild_encodings_empty)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("Failed checks:", FAIL)
    sys.exit(1)
print("All smoke checks passed.")
