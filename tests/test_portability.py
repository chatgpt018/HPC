"""Cross-directory behavior that matters after extracting the project elsewhere."""
from pathlib import Path

from marketintel.storage import data_directory, home


def test_relative_data_directory_is_project_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', 'storage/portable-check')
    expected = Path(__file__).resolve().parents[1] / 'storage' / 'portable-check'
    assert data_directory() == expected
    assert not expected.exists()  # Diagnostics must not create runtime state.


def test_absolute_data_directory_is_supported(monkeypatch, tmp_path):
    target = tmp_path / 'custom-data'
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(target))
    assert home() == target
    assert target.is_dir()
