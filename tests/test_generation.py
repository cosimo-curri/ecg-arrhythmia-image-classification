"""Exercise the image generator with fake WFDB annotations and a real render."""

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from ecg_classification.data import generate_record_images, load_images


def test_generate_one_original_and_nine_crops(monkeypatch, tmp_path):
    dataset = tmp_path / "mit-bih-arrhythmia-database-1.0.0"
    dataset.mkdir()
    for suffix in ("dat", "hea", "atr"):
        (dataset / f"100.{suffix}").write_text("synthetic fixture")
    times = np.arange(600)
    signal = np.column_stack((np.sin(times / 13), np.cos(times / 13)))
    annotations = SimpleNamespace(sample=np.array([20, 150, 300, 500]),
                                  symbol=["N", "/", "E", "N"])
    fake_wfdb = SimpleNamespace(rdsamp=lambda path: (signal, {}),
                                rdann=lambda path, extension: annotations)
    monkeypatch.setitem(sys.modules, "wfdb", fake_wfdb)
    output = tmp_path / "dataset_images"
    count = generate_record_images(dataset, output, "100", "train", maximum_beats=1)
    assert count == 1
    base = load_images(output, "train", ["100"])
    all_images = load_images(output, "train", ["100"], include_crops=True)
    assert len(base) == 1 and len(all_images) == 10
    assert base[0].label == "PACE"
    assert len({image.path.name for image in all_images}) == 10
    assert all(cv2.imread(str(item.path), 0).shape == (128, 128) for item in all_images)


def test_regeneration_does_not_reuse_stale_crops(monkeypatch, tmp_path):
    """A fresh run must honor the new beat limit and crop setting."""
    dataset = tmp_path / "records"
    dataset.mkdir()
    for suffix in ("dat", "hea", "atr"):
        (dataset / f"100.{suffix}").write_text("synthetic fixture")
    times = np.arange(600)
    signal = np.column_stack((np.sin(times / 13), np.cos(times / 13)))
    annotations = SimpleNamespace(sample=np.array([20, 150, 300, 500]),
                                  symbol=["N", "/", "E", "N"])
    monkeypatch.setitem(sys.modules, "wfdb", SimpleNamespace(
        rdsamp=lambda prefix: (signal, {}),
        rdann=lambda prefix, extension: annotations,
    ))
    out = tmp_path / "generated"
    assert generate_record_images(dataset, out, "100", "train", augmentation=True,
                                  maximum_beats=2) == 2
    assert len(load_images(out, "train", ["100"], include_crops=True)) == 20
    assert generate_record_images(dataset, out, "100", "train", augmentation=False,
                                  maximum_beats=1) == 1
    assert len(load_images(out, "train", ["100"], include_crops=True)) == 1
    assert len(load_images(out, "train", ["100"])) == 1
