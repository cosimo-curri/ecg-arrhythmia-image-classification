"""Tests for record separation and ECG image transformations."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ecg_classification.data import (
    CLASS_NAMES, SYMBOL_TO_CLASS, ImageSample, crop_nine, image_arrays,
    load_images, parse_image_name, render_segment, split_records, split_class_coverage,
    record_class_labels,
)


def test_record_split_is_reproducible_and_disjoint():
    records = [str(number) for number in range(100, 148)]
    first = split_records(records)
    second = split_records(records[::-1])
    assert first == second
    assert [len(first[key]) for key in ("train", "validation", "test")] == [33, 7, 8]
    groups = [set(values) for values in first.values()]
    assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert set().union(*groups) == set(records)


def test_split_rejects_duplicate_or_too_few_records():
    with pytest.raises(ValueError):
        split_records(["100", "100", "101"])
    with pytest.raises(ValueError):
        split_records(["100", "101"])


def test_wfdb_mapping_preserves_class_indices_but_corrects_meaning():
    assert len(CLASS_NAMES) == 8
    assert SYMBOL_TO_CLASS["/"] == "PACE"
    assert SYMBOL_TO_CLASS["E"] == "VESC"
    assert SYMBOL_TO_CLASS["A"] == "APC"
    assert "." not in SYMBOL_TO_CLASS  # Not a valid beat annotation in the dataset.


def test_rare_classes_remain_train_only_when_supported_by_one_record():
    # 48 synthetic recordings; 207 is the only source of flutter waves.
    records = [str(i) for i in range(100, 147)] + ["207"]
    metadata = {name: {"NOR", "PVC"} for name in records}
    metadata["207"] = {"LBB", "RBB", "VFW", "VESC"}
    for index in range(110, 114):
        metadata[str(index)].add("LBB")
    for index in range(114, 120):
        metadata[str(index)].add("RBB")
    for index in range(120, 126):
        metadata[str(index)].add("PACE")
    for index in range(126, 133):
        metadata[str(index)].add("APC")
    metadata["133"].add("VESC")
    first = split_records(records, record_labels=metadata)
    assert first == split_records(records[::-1], record_labels=metadata)
    assert set().union(*(set(v) for v in first.values())) == set(records)
    assert "207" in first["train"]
    assert all(set(first[a]).isdisjoint(first[b]) for a,b in
               (("train", "validation"), ("train", "test"), ("validation", "test")))
    coverage = split_class_coverage(first, metadata)
    assert coverage["VFW"] == {"train": True, "validation": False, "test": False}
    assert all(c["train"] for c in coverage.values())
    for label in ("PACE", "LBB", "RBB", "APC"):
        assert coverage[label]["validation"] and coverage[label]["test"]
    assert coverage["VESC"]["validation"] or coverage["VESC"]["test"]


def test_record_split_validates_provided_metadata():
    records = ["100", "101", "102", "103", "104", "105"]
    with pytest.raises(ValueError, match="exactly"):
        split_records(records, record_labels={"100": {"NOR"}})
    with pytest.raises(ValueError, match="unknown"):
        split_records(records, record_labels={r: {"NOT_REAL"} if r == "105" else {"NOR"}
                                               for r in records})


def test_record_class_scan_reads_wfdb_annotations(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    annotations = SimpleNamespace(
        sample=[0, 100, 250, 400, 550, 700, 850, 1000],
        symbol=["+", "N", ".", "V", "E", "!", "N", "+"],
    )
    monkeypatch.setitem(sys.modules, "wfdb", SimpleNamespace(
        rdann=lambda prefix, extension: annotations,
    ))
    assert record_class_labels(tmp_path, ["100"]) == {
        "100": {"NOR", "PVC", "VESC", "VFW"}
    }


def test_nine_crops_match_original_offsets():
    image = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128).astype("uint8")
    crops = crop_nine(image)
    assert len(crops) == 9
    assert all(crop.shape == (128, 128) for crop in crops)
    assert np.array_equal(crops[0], cv2.resize(image[:96, :96], (128, 128)))
    assert np.array_equal(crops[4], cv2.resize(image[16:112, 16:112], (128, 128)))
    assert np.array_equal(crops[8], cv2.resize(image[32:, 32:], (128, 128)))


def test_file_parsing_and_record_filtered_loading(tmp_path):
    for split, record in (("train", "100"), ("validation", "101"), ("test", "102")):
        folder = tmp_path / split / "PACE"
        folder.mkdir(parents=True)
        for index in (0, 1):
            assert cv2.imwrite(str(folder / f"PACE_{record}_120_200{index}.png"),
                               np.full((128, 128), 175, dtype="uint8"))
    base = parse_image_name(tmp_path / "train" / "PACE" / "PACE_100_120_2000.png")
    assert (base.record, base.label, base.variant) == ("100", "PACE", 0)
    assert len(load_images(tmp_path, "train", ["100"])) == 1
    assert len(load_images(tmp_path, "train", ["100"], include_crops=True)) == 2
    assert not load_images(tmp_path, "train", ["101"])
    X, Y = image_arrays([base], augment=False)
    assert X.shape == (1, 64, 64, 1)
    assert Y.shape == (1, 8)
    assert Y[0, 2] == 1
    assert X.dtype == np.float32
    assert Y.dtype == np.float32
    assert np.isclose(X[0, 0, 0, 0], 175 / 255)
    with pytest.raises(ValueError):
        parse_image_name(Path("PAB_100_bad.png"))


def test_rendered_segment_matches_independent_original_plot(tmp_path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    times = np.arange(600)
    signal = np.column_stack((np.sin(times / 21), np.cos(times / 21)))
    start, end = 80, 170
    actual = render_segment(signal, start, end, tmp_path / "new.png")
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(start, end), signal[start:end, 0], color="black")
    ax.axis("off")
    original_path = tmp_path / "original.png"
    canvas.print_figure(str(original_path), dpi=100, bbox_inches="tight", pad_inches=0)
    original = cv2.resize(cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE),
                          (128, 128), interpolation=cv2.INTER_LANCZOS4)
    assert np.array_equal(actual, original)
    assert np.array_equal(cv2.imread(str(tmp_path / "new.png"), 0), original)


def test_shared_subject_records_are_never_split():
    """MIT-BIH 201 and 202 originate from the same subject (PhysioNet)."""
    records = [str(i) for i in range(100, 144)] + ["201", "202", "207", "234"]
    labels = {record: {"NOR", "PVC"} for record in records}
    labels["207"].add("VFW")
    for seed in (0, 1, 17, 123):
        for metadata in (None, labels):
            result = split_records(records, seed=seed, record_labels=metadata)
            assert [len(result[key]) for key in ("train", "validation", "test")] == [33, 7, 8]
            assert sum("201" in members for members in result.values()) == 1
            assert all(("201" in members) == ("202" in members)
                       for members in result.values())
            if metadata is not None:
                assert "207" in result["train"]
            assert result == split_records(records[::-1], seed=seed, record_labels=metadata)


def test_shared_subject_group_validation():
    with pytest.raises(ValueError, match="overlap"):
        split_records(["100", "101", "102", "103", "104", "105"],
                      shared_subject_groups=(("100", "101"), ("101", "102")))
