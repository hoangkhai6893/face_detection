"""
Unit tests for PersonManager (CRUD operations).
"""

import cv2
import pytest

from training_manager import PersonManager


@pytest.fixture
def manager(tmp_dataset):
    return PersonManager(dataset_path=str(tmp_dataset))


class TestListPersons:
    def test_list_returns_two_persons(self, manager):
        persons = manager.list_persons()
        names = {p.name for p in persons}
        assert names == {"Alice", "Bob"}

    def test_list_empty_dataset(self, tmp_path):
        mgr = PersonManager(dataset_path=str(tmp_path))
        assert mgr.list_persons() == []

    def test_image_count_correct(self, manager):
        persons = {p.name: p for p in manager.list_persons()}
        assert persons["Alice"].image_count == 3
        assert persons["Bob"].image_count == 3

    def test_augmented_count_zero_initially(self, manager):
        for p in manager.list_persons():
            assert p.augmented_count == 0


class TestCreatePerson:
    def test_create_new_person(self, manager, tmp_dataset):
        path = manager.create_person("Charlie")
        assert path.is_dir()
        assert (tmp_dataset / "Charlie").is_dir()

    def test_create_already_exists_is_idempotent(self, manager, tmp_dataset):
        # Should not raise; folder already exists
        path = manager.create_person("Alice")
        assert path.is_dir()

    def test_create_invalid_name_raises(self, manager):
        with pytest.raises(ValueError):
            manager.create_person("../evil")

    def test_create_empty_name_raises(self, manager):
        with pytest.raises(ValueError):
            manager.create_person("   ")

    def test_exists_returns_true_after_create(self, manager):
        manager.create_person("Dave")
        assert manager.exists("Dave")

    def test_exists_returns_false_for_unknown(self, manager):
        assert not manager.exists("Nobody")


class TestDeletePerson:
    def test_delete_removes_folder(self, manager, tmp_dataset):
        manager.delete_person("Alice", rebuild=False)
        assert not (tmp_dataset / "Alice").exists()

    def test_delete_returns_true_on_success(self, manager):
        result = manager.delete_person("Alice", rebuild=False)
        assert result is True

    def test_delete_nonexistent_returns_false(self, manager):
        result = manager.delete_person("Nobody", rebuild=False)
        assert result is False


class TestResetPerson:
    def test_reset_clears_images(self, manager, tmp_dataset):
        manager.reset_person("Alice", rebuild=False)
        remaining = list((tmp_dataset / "Alice").iterdir())
        assert remaining == [], "Folder should be empty after reset"

    def test_reset_keeps_folder(self, manager, tmp_dataset):
        manager.reset_person("Bob", rebuild=False)
        assert (tmp_dataset / "Bob").is_dir()

    def test_reset_returns_correct_count(self, manager):
        count = manager.reset_person("Alice", rebuild=False)
        assert count == 3   # 3 images were created in tmp_dataset fixture

    def test_reset_nonexistent_returns_zero(self, manager):
        count = manager.reset_person("Nobody", rebuild=False)
        assert count == 0


class TestGetCounts:
    def test_get_image_count(self, manager):
        assert manager.get_image_count("Alice") == 3

    def test_get_image_count_excludes_aug(self, manager, tmp_dataset, synthetic_face_image):
        # Write a fake augmented file
        aug_path = tmp_dataset / "Alice" / "Alice_0_aug_blur.jpg"
        cv2.imwrite(str(aug_path), synthetic_face_image)
        # Original count should still be 3
        assert manager.get_image_count("Alice") == 3
        # Augmented count should be 1
        assert manager.get_augmented_count("Alice") == 1

    def test_get_image_count_missing_person(self, manager):
        assert manager.get_image_count("Nobody") == 0
