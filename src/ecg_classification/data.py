"""MIT-BIH ECG segmentation, image creation, and record-disjoint data loading.

This module implements segmentation, plotting, resize interpolation, crop
coordinates, and normalization for the ECG-to-image pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from importlib.resources import files
import hashlib
import random
from typing import Iterator, Mapping, Sequence

import cv2
import numpy as np

# Output indices are shared across preprocessing, training and evaluation.
# WFDB '/' denotes paced beats; 'E' denotes ventricular escape beats.
CLASS_NAMES = ("NOR", "PVC", "PACE", "LBB", "RBB", "APC", "VFW", "VESC")
SYMBOL_TO_CLASS = {
    "N": "NOR",
    "V": "PVC",
    "/": "PACE",
    "L": "LBB",
    "R": "RBB",
    "A": "APC",
    "!": "VFW",
    "E": "VESC",
}
CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASS_NAMES)}
# PhysioNet documents recordings 201 and 202 as belonging to one subject.
# A record-disjoint split must keep these two recordings together to avoid
# subject-level data leakage.
MIT_BIH_SHARED_SUBJECT_GROUPS = (("201", "202"),)


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str
    record: str
    variant: int



def _primary_file_hashes() -> dict[str, str]:
    """Pinned SHA-256 checksums for the 48 official MIT-BIH record triplets.

    The reference hashes were taken from the version 1.0.0 SHA256SUMS.txt
    included in the source dataset. Unlike a manifest inside a downloaded
    archive, these package-owned hashes cannot be replaced by that archive.
    """
    manifest = files("ecg_classification").joinpath(
        "resources/mit_bih_primary_sha256.txt"
    )
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, filename = line.split(maxsplit=1)
        filename = filename.strip()
        if filename in expected:
            raise ValueError(f"Duplicate file in bundled MIT-BIH manifest: {filename}")
        expected[filename] = digest.lower()
    record_names = {name[:-4] for name in expected}
    if (len(expected) != 144 or len(record_names) != 48 or
            any({f"{record}.{ext}" for ext in ("atr", "dat", "hea")}
                - expected.keys() for record in record_names)):
        raise ValueError("Bundled MIT-BIH checksum manifest is incomplete")
    return expected


def verify_mit_bih_records(dataset_directory: Path) -> None:
    """Verify all 144 primary MIT-BIH v1.0.0 files against pinned hashes.

    This also validates manually supplied datasets before preprocessing. It
    cannot authenticate the provenance of the university source archive.
    """
    directory = Path(dataset_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"MIT-BIH directory does not exist: {directory}")
    expected = _primary_file_hashes()
    missing = sorted(name for name in expected if not (directory / name).is_file())
    if missing:
        raise ValueError(f"Missing MIT-BIH record files: {', '.join(missing[:5])}")
    for filename, expected_digest in expected.items():
        digest = hashlib.sha256()
        with (directory / filename).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"SHA-256 mismatch for MIT-BIH file: {filename}")


def record_ids(dataset_directory: Path) -> list[str]:
    """Discover complete three-digit MIT-BIH record triplets."""
    directory = Path(dataset_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"MIT-BIH directory does not exist: {directory}")
    records = sorted(
        file.stem for file in directory.glob("[0-9][0-9][0-9].dat")
        if (directory / (file.stem + ".hea")).is_file()
        and (directory / (file.stem + ".atr")).is_file()
    )
    if not records:
        raise ValueError(f"No complete MIT-BIH record triplets found in {directory}")
    return records


def record_class_labels(dataset_directory: Path, records: Sequence[str]) -> dict[str, set[str]]:
    """Discover the output classes available in each WFDB annotation file.

    An annotation is eligible only when the same 20-sample margins used by the
    image generator leave a nonempty segment. A label's presence does not
    guarantee a sufficient number of examples for scientific evaluation.
    """
    import wfdb

    directory = Path(dataset_directory)
    labels_by_record: dict[str, set[str]] = {}
    for record in records:
        annotations = wfdb.rdann(str(directory / record), extension="atr")
        classes: set[str] = set()
        for i in range(1, len(annotations.sample) - 1):
            label = SYMBOL_TO_CLASS.get(annotations.symbol[i])
            if label is None:
                continue
            start = int(annotations.sample[i - 1]) + 20
            end = int(annotations.sample[i + 1]) - 20
            if end > start:
                classes.add(label)
        labels_by_record[record] = classes
    return labels_by_record


def split_class_coverage(
    partitions: Mapping[str, Sequence[str]], record_labels: Mapping[str, set[str]],
) -> dict[str, dict[str, bool]]:
    """Report class presence in each record-disjoint partition."""
    return {
        label: {
            split: any(label in record_labels[record] for record in partitions[split])
            for split in ("train", "validation", "test")
        }
        for label in CLASS_NAMES
    }


def split_records(
    records: Sequence[str], seed: int = 0, train_fraction: float = 0.70,
    *, record_labels: Mapping[str, set[str]] | None = None,
    max_attempts: int = 4096,
    shared_subject_groups: Sequence[Sequence[str]] = MIT_BIH_SHARED_SUBJECT_GROUPS,
) -> dict[str, list[str]]:
    """Split recordings into approximately 70/15/15 train/validation/test.

    Recordings known to belong to the same patient are kept together. For
    MIT-BIH these are 201 and 202; groups absent from a subset are ignored.
    When class metadata is provided, deterministic candidate splits favor
    coverage of all classes in training and of rare classes in holdouts where
    feasible. Recordings are never duplicated to manufacture test examples.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(records) < 3 or len(set(records)) != len(records):
        raise ValueError("At least three unique record IDs are required")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    items = sorted(records)
    record_set = set(items)
    groups = [set(group) for group in shared_subject_groups if set(group) <= record_set]
    if any(len(group) < 2 for group in groups):
        raise ValueError("shared_subject_groups must contain at least two distinct records")
    if sum(map(len, groups)) != len(set().union(*groups)):
        raise ValueError("shared_subject_groups must not overlap")
    train_count = int(len(items) * train_fraction)
    held_out_count = len(items) - train_count
    validation_count = held_out_count // 2
    if train_count < 1 or validation_count < 1 or held_out_count - validation_count < 1:
        raise ValueError("Partition produces empty train, validation or test subset")

    if record_labels is not None:
        if set(record_labels) != set(items):
            raise ValueError("record_labels must cover exactly the requested records")
        if any(set(labels) - set(CLASS_NAMES) for labels in record_labels.values()):
            raise ValueError("record_labels contains unknown classes")
        frequency = Counter(label for labels in record_labels.values() for label in labels)
    else:
        frequency = Counter()

    best: dict[str, list[str]] | None = None
    best_score = -1
    for attempt in range(max_attempts if groups or record_labels is not None else 1):
        shuffled = items.copy()
        random.Random(seed + attempt).shuffle(shuffled)
        partitions = {
            "train": shuffled[:train_count],
            "validation": shuffled[train_count:train_count + validation_count],
            "test": shuffled[train_count + validation_count:],
        }
        if groups:
            # Reject partitions that leak recordings from the same subject
            # across train, validation or test. Preserve exact subset sizes.
            partition_by_record = {
                record: subset for subset, members in partitions.items() for record in members
            }
            if any(len({partition_by_record[record] for record in group}) != 1
                   for group in groups):
                continue
        if record_labels is None or not frequency:
            return partitions
        coverage = split_class_coverage(partitions, record_labels)
        train_present = sum(coverage[label]["train"] for label in frequency)
        held_out_present = sum(
            coverage[label]["validation"] or coverage[label]["test"]
            for label, count in frequency.items() if count >= 2
        )
        evaluation_present = sum(
            coverage[label]["validation"] + coverage[label]["test"]
            for label, count in frequency.items() if count >= 3
        )
        score = train_present * 1000 + held_out_present * 100 + evaluation_present * 10
        if score > best_score:
            best, best_score = partitions, score
        if (
            train_present == len(frequency)
            and held_out_present == sum(count >= 2 for count in frequency.values())
            and evaluation_present == 2 * sum(count >= 3 for count in frequency.values())
        ):
            return partitions

    if best is None:
        raise ValueError("No feasible split found with the shared-subject constraints")
    missing_training = [
        label for label in frequency
        if not split_class_coverage(best, record_labels)[label]["train"]
    ]
    if missing_training:
        raise ValueError(
            "Cannot assign all present classes to training; missing: "
            + ", ".join(sorted(missing_training))
        )
    # A split may still lack a rare class if it is supported by too few records
    # or cannot be spread across partitions without violating record separation.
    return best


def crop_nine(image: np.ndarray, size: tuple[int, int] = (128, 128)) -> list[np.ndarray]:
    """Return a 3 x 3 grid of 96 x 96 crops."""
    if image.shape != (128, 128):
        raise ValueError("The nine-crop algorithm requires a 128 x 128 grayscale image")
    return [
        cv2.resize(image[y:y + 96, x:x + 96], size)
        for y in (0, 16, 32)
        for x in (0, 16, 32)
    ]


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def render_segment(signal: np.ndarray, start: int, end: int, destination: Path,
                   size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Use Matplotlib Agg to render ECG signal channel 0."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    if start < 0 or end > signal.shape[0] or end <= start:
        raise ValueError("ECG segment indices are outside the signal range")
    figure = Figure()
    canvas = FigureCanvas(figure)
    axes = figure.add_subplot(111)
    axes.plot(np.arange(start, end), signal[start:end, 0], color="black")
    axes.axis("off")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.print_figure(str(destination), dpi=100, bbox_inches="tight", pad_inches=0)
        grayscale = cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise OSError(f"Could not read rendered image: {destination}")
        grayscale = cv2.resize(grayscale, size, interpolation=cv2.INTER_LANCZOS4)
        _write_image(destination, grayscale)
        return grayscale
    finally:
        figure.clear()


def generate_record_images(
    dataset_directory: Path,
    output_directory: Path,
    record: str,
    split: str,
    *,
    augmentation: bool = True,
    maximum_beats: int | None = None,
    ignore_samples: int = 20,
) -> int:
    """Generate one 128-pixel image per annotated beat and optionally nine crops.

    The caller supplies a record-disjoint partition and whether to generate
    additional cropped variants for each accepted annotation.
    """
    import wfdb

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split}")
    if record not in record_ids(dataset_directory):
        raise ValueError(f"Unknown or incomplete MIT-BIH record: {record}")
    if maximum_beats is not None and maximum_beats <= 0:
        raise ValueError("maximum_beats must be positive")

    prefix = str(Path(dataset_directory) / record)
    signal, _metadata = wfdb.rdsamp(prefix)
    annotations = wfdb.rdann(prefix, extension="atr")
    # Rerunning with fewer beats or without crops must not silently reuse PNGs
    # from an earlier, larger preparation. Only the selected record/split is reset.
    for label in CLASS_NAMES:
        folder = Path(output_directory) / split / label
        if folder.is_dir():
            for previous in folder.glob(f"{label}_{record}_*.png"):
                previous.unlink()
    count = 0
    for i in range(1, len(annotations.sample) - 1):
        symbol = annotations.symbol[i]
        if symbol not in SYMBOL_TO_CLASS:
            continue
        start = int(annotations.sample[i - 1]) + ignore_samples
        end = int(annotations.sample[i + 1]) - ignore_samples
        if start < 0 or end > len(signal) or end <= start:
            continue
        label = SYMBOL_TO_CLASS[symbol]
        folder = Path(output_directory) / split / label
        name = f"{label}_{record}_{start}_{end}0.png"
        full_path = folder / name
        image = render_segment(signal, start, end, full_path)
        if augmentation:
            for variant, crop in enumerate(crop_nine(image), start=1):
                _write_image(folder / f"{label}_{record}_{start}_{end}{variant}.png", crop)
        count += 1
        if maximum_beats is not None and count >= maximum_beats:
            break
    return count


def parse_image_name(path: Path) -> ImageSample:
    """Parse labels and record IDs from generated ECG image filenames."""
    path = Path(path)
    try:
        label, record, start, end_and_variant = path.stem.split("_")
        if label not in CLASS_TO_INDEX or len(record) != 3 or not record.isdigit():
            raise ValueError("Unknown class or invalid record")
        if not start.isdigit() or len(end_and_variant) < 2 or not end_and_variant.isdigit():
            raise ValueError("Invalid segment boundaries")
        variant = int(end_and_variant[-1])
        if variant not in range(10):
            raise ValueError("Invalid crop index")
    except ValueError as exc:
        raise ValueError(f"Invalid ECG filename: {path.name}") from exc
    return ImageSample(path, label, record, variant)


def load_images(
    output_directory: Path,
    split: str,
    allowed_records: Sequence[str],
    *,
    include_crops: bool = False,
) -> list[ImageSample]:
    """Load only images belonging to the requested split and record IDs."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split: {split}")
    allowed = set(allowed_records)
    images: list[ImageSample] = []
    for label in CLASS_NAMES:
        for path in sorted((Path(output_directory) / split / label).glob("*.png")):
            sample = parse_image_name(path)
            if sample.label != label:
                raise ValueError(f"Folder and filename label disagree: {path}")
            if sample.record in allowed and (include_crops or sample.variant == 0):
                images.append(sample)
    return images


def image_arrays(
    samples: Sequence[ImageSample],
    *,
    size: tuple[int, int] = (64, 64),
    augment: bool = False,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and one-hot encode images, optionally applying augmentations."""
    if augment:
        import imutils

    rng = rng if rng is not None else random.Random(0)
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sample in samples:
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load ECG image: {sample.path}")
        if augment and rng.uniform(0, 1) < 0.30:
            image = imutils.rotate(image, rng.randint(-180, 180))
        if augment and rng.uniform(0, 1) < 0.30:
            image = cv2.flip(image, rng.randint(-1, 1))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        image = image.astype("float32") / np.float32(255.0)
        images.append(image.reshape((size[1], size[0], 1)))
        target = np.zeros((len(CLASS_NAMES),), dtype="float32")
        target[CLASS_TO_INDEX[sample.label]] = 1.0
        labels.append(target)
    return (
        np.stack(images) if images else np.empty((0, size[1], size[0], 1), dtype="float32"),
        np.stack(labels) if labels else np.empty((0, len(CLASS_NAMES)), dtype="float32"),
    )


def batches(
    samples: Sequence[ImageSample], batch_size: int = 32,
    *, augment: bool = False, seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream batches to TensorFlow without loading a million images into RAM."""
    if batch_size <= 0 or not samples:
        raise ValueError("batch_size must be positive and samples nonempty")
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    while True:
        if augment:
            rng.shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            batch = [samples[j] for j in indices[offset:offset + batch_size]]
            yield image_arrays(batch, augment=augment, rng=rng)


def download_mit_bih(
    destination: Path = Path("mit-bih-arrhythmia-database-1.0.0"),
    url: str = "https://physionet.org/content/mitdb/get-zip/1.0.0/",
) -> Path:
    """Download the 48 MIT-BIH triplets and verify each pinned SHA-256 digest.

    Extracts only the official record filenames into a temporary directory.
    A fully verified directory is moved into place; a failed operation leaves
    no partial dataset behind. The default source is PhysioNet v1.0.0.
    """
    import tempfile
    from urllib.request import urlopen
    from zipfile import ZipFile

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _primary_file_hashes()
    archive_root = "mit-bih-arrhythmia-database-1.0.0"

    with tempfile.TemporaryDirectory(prefix=".mit-bih-", dir=destination.parent) as work:
        staging = Path(work) / "dataset"
        staging.mkdir()
        with tempfile.TemporaryFile() as archive_file:
            with urlopen(url, timeout=60) as response:
                total_downloaded = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    total_downloaded += len(chunk)
                    if total_downloaded > 512 * 1024 * 1024:
                        raise ValueError("MIT-BIH ZIP exceeds the download safety limit")
                    archive_file.write(chunk)
            archive_file.seek(0)
            with ZipFile(archive_file) as archive:
                seen: set[str] = set()
                for member in archive.infolist():
                    parts = Path(member.filename).parts
                    if member.is_dir() or len(parts) != 2 or parts[0] != archive_root:
                        continue
                    filename = parts[1]
                    if filename not in expected:
                        continue
                    if filename in seen:
                        raise ValueError(f"Duplicate MIT-BIH archive member: {filename}")
                    if member.file_size > 32 * 1024 * 1024:
                        raise ValueError(f"Unexpectedly large MIT-BIH record file: {filename}")
                    seen.add(filename)
                    checksum = hashlib.sha256()
                    with archive.open(member) as source, (staging / filename).open("wb") as target:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            checksum.update(chunk)
                            target.write(chunk)
                    if checksum.hexdigest() != expected[filename]:
                        raise ValueError(f"MIT-BIH SHA-256 mismatch: {filename}")
                missing = expected.keys() - seen
                if missing:
                    raise ValueError(
                        f"Missing MIT-BIH record files in archive: {', '.join(sorted(missing)[:5])}"
                    )
        if destination.exists():
            raise FileExistsError(f"Destination appeared during download: {destination}")
        staging.rename(destination)
    return destination
