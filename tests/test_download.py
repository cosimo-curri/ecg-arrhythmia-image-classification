"""Integrity, completeness, and safe extraction of MIT-BIH ZIP inputs."""

import hashlib
import io
import zipfile
import warnings

import pytest

from ecg_classification import data


@pytest.fixture
def archive_data(monkeypatch):
    # Use the 144 real *filenames*, but tiny deterministic synthetic contents.
    # Only this fixture's checksums are replaced; production hashes stay pinned.
    filenames = sorted(data._primary_file_hashes())
    contents = {filename: f"fixture for {filename}".encode() for filename in filenames}
    digests = {filename: hashlib.sha256(content).hexdigest()
               for filename, content in contents.items()}
    monkeypatch.setattr(data, "_primary_file_hashes", lambda: digests)
    return contents


def make_archive(contents, *, corrupt=None, omit=None, duplicate=None):
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        for filename, payload in contents.items():
            if filename == omit:
                continue
            archive.writestr(
                f"mit-bih-arrhythmia-database-1.0.0/{filename}",
                b"bad content" if corrupt == filename else payload,
            )
        if duplicate:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Duplicate name", category=UserWarning)
                archive.writestr(
                    f"mit-bih-arrhythmia-database-1.0.0/{duplicate}", contents[duplicate]
                )
        archive.writestr("mit-bih-arrhythmia-database-1.0.0/../malicious.txt", "BAD")
        archive.writestr("outside.txt", "BAD")
        archive.writestr("mit-bih-arrhythmia-database-1.0.0/999.dat", "BAD")
    return archive_buffer.getvalue()


def test_download_checks_hashes_record_names_and_ignores_unrelated_files(
    archive_data, monkeypatch, tmp_path,
):
    content = make_archive(archive_data)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(content))
    destination = data.download_mit_bih(tmp_path / "records")
    assert len(list(destination.glob("*.dat"))) == 48
    assert len(list(destination.glob("*.hea"))) == 48
    assert len(list(destination.glob("*.atr"))) == 48
    assert not (destination / "999.dat").exists()
    assert not (tmp_path / "malicious.txt").exists()
    data.verify_mit_bih_records(destination)


@pytest.mark.parametrize("problem", ["corrupt", "omit", "duplicate"])
def test_download_rejects_bad_archives_and_cleans_up(
    archive_data, monkeypatch, tmp_path, problem,
):
    filename = "100.atr"
    content = make_archive(archive_data, **{problem: filename})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(content))
    destination = tmp_path / "records"
    with pytest.raises(ValueError):
        data.download_mit_bih(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".mit-bih-*"))


def test_verifier_rejects_tampering_and_missing_files(archive_data, tmp_path):
    folder = tmp_path / "records"
    folder.mkdir()
    for name, contents in archive_data.items():
        (folder / name).write_bytes(contents)
    data.verify_mit_bih_records(folder)
    (folder / "100.atr").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.verify_mit_bih_records(folder)
    (folder / "100.atr").unlink()
    with pytest.raises(ValueError, match="Missing"):
        data.verify_mit_bih_records(folder)


def test_download_network_failure_does_not_leave_partial_data(monkeypatch, tmp_path):
    def fail(*_args, **_kwargs):
        raise OSError("offline")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    destination = tmp_path / "records"
    with pytest.raises(OSError, match="offline"):
        data.download_mit_bih(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".mit-bih-*"))
