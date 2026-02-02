"""Tests for object name extraction from task instructions.

All tests are standalone. No LIBERO or MuJoCo needed.
"""

import pytest

from sketch_anything.registry.extractor import extract_object_names


class TestExtractObjectNames:
    def test_pick_and_place(self):
        names = extract_object_names(
            "pick up the red block and place it in the blue bowl"
        )
        assert names == ["red block", "blue bowl"]

    def test_put_on(self):
        names = extract_object_names("put the mug on the plate")
        assert names == ["mug", "plate"]

    def test_open(self):
        names = extract_object_names("open the top drawer")
        assert names == ["top drawer"]

    def test_push(self):
        names = extract_object_names("push the red block")
        assert names == ["red block"]

    def test_pick_up_only(self):
        names = extract_object_names("pick up the cream cheese")
        assert names == ["cream cheese"]

    def test_turn_on(self):
        names = extract_object_names("turn on the stove")
        assert names == ["stove"]

    def test_place_into(self):
        names = extract_object_names("place the butter into the bowl")
        assert names == ["butter", "bowl"]

    def test_stack(self):
        names = extract_object_names("stack the blue block on top of the red block")
        assert names == ["blue block", "red block"]

    def test_deduplication(self):
        """Same object referenced twice should appear once."""
        names = extract_object_names("move the mug to the mug")
        assert names == ["mug"]

    def test_case_insensitive(self):
        names = extract_object_names("Pick Up The Red Block")
        assert names == ["Red Block"]

    def test_complex_names(self):
        names = extract_object_names(
            "pick up the orange juice and place it on the tray"
        )
        assert names == ["orange juice", "tray"]
