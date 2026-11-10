"""Local interface-action checks; no browser, AWS, MPI runtime or LLM required."""
import json
from pathlib import Path

from marketintel.ui_actions import (
    upload, ask, run_analysis, inspect_result, result_chart, show_benchmark,
)
from marketintel.storage import home, write_json


def _market_csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text('topic,summary\n' + f'"{name}","{text}"' + '\n', encoding='utf-8')
    return path


def test_user_actions_ingest_retrieve_and_calculate(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    report_files = [
        _market_csv(tmp_path, 'ev.csv', 'Electric vehicle battery manufacturing and energy storage.'),
        _market_csv(tmp_path, 'solar.csv', 'Solar rooftop energy panels and renewable electricity.'),
        _market_csv(tmp_path, 'chips.csv', 'Semiconductor devices microprocessors and chip production.'),
    ]
    log, rows = upload([str(x) for x in report_files], 'Synthetic sample')
    assert 'Indexed' in log
    assert len(rows) == 3
    assert all(row[2] >= 1 for row in rows)

    response, evidence = ask('Which report mentions solar rooftop electricity?', False)
    assert 'Evidence-only' in response
    assert evidence and any('solar.csv' in str(row) for row in evidence)

    info, data, metadata = run_analysis('Exact similarity · top-k', 'Sequential', 2, 2, 'hashing')
    assert '**Sequential local computation**' in info
    assert 'MPI not used' in info
    assert data and len(json.loads(metadata)['source_documents']) == 3
    assert (home() / 'results').exists()

    result_name = next((home() / 'results').glob('*.json')).name
    description, saved_rows, receipt = inspect_result(result_name)
    assert 'sequential' in description.lower()
    assert saved_rows
    assert 'input_sha256' in receipt
    assert result_chart(result_name) is not None

    cluster_info, clusters, cluster_meta = run_analysis('K-Means clustering', 'Sequential', 2, 2, 'hashing')
    assert 'Sequential' in cluster_info
    assert clusters and json.loads(cluster_meta)['algorithm']

    benchmark_note, empty_chart = show_benchmark(None)
    assert 'No real MPI benchmark' in benchmark_note
    assert empty_chart is None


def test_failed_real_mpi_request_is_not_displayed_as_completed(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    for i, topic in enumerate(['solar', 'battery']):
        file = _market_csv(tmp_path, f'file{i}.csv', f'Example market report about {topic}.')
        upload([str(file)], 'Synthetic sample')

    def no_mpi(*args, **kwargs):
        raise RuntimeError('MPI launcher not available in this environment')

    monkeypatch.setattr('marketintel.ui_actions.run_local_mpi', no_mpi)
    status, table, metadata = run_analysis('Exact similarity · top-k', 'Real MPI', 4, 1, 'hashing')
    assert 'not completed' in status.lower()
    assert 'MPI launcher not available' in status
    assert table == [] and metadata == ''


def test_single_document_does_not_create_empty_similarity_result(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    file = _market_csv(tmp_path, 'one.csv', 'Synthetic solar supply chain practice report.')
    upload([str(file)], 'Synthetic sample')

    status, table, metadata = run_analysis('Exact similarity · top-k', 'Sequential', 2, 3, 'hashing')
    assert 'at least two indexed documents' in status
    assert table == [] and metadata == ''
    assert not list((home() / 'results').glob('*.json'))


def test_invalid_execution_mode_cannot_be_reported_as_sequential(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    status, table, metadata = run_analysis('Exact similarity · top-k', 'Unknown mode', 2, 1, 'hashing')
    assert 'not completed' in status.lower()
    assert 'Choose Sequential or Real MPI' in status
    assert table == [] and metadata == ''
    assert not (home() / 'datasets' / 'uploaded_reports.npz').exists()


def test_benchmark_view_labels_synthetic_input(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    write_json(home() / 'benchmarks' / 'synthetic_benchmark_2p.json', {
        'dataset_kind': 'SYNTHETIC NUMERICAL BENCHMARK INPUT',
        'operation': 'similarity', 'documents': 3, 'mpi_processes': 2,
        'median_speedup': 0.1,
        'runs': [{'sequential_wall_seconds': 0.1, 'mpi_end_to_end_wall_seconds': 1.0}],
    })
    summary, chart = show_benchmark('synthetic_benchmark_2p.json')
    assert 'SYNTHETIC NUMERICAL BENCHMARK INPUT' in summary
    assert '0.100×' in summary and chart is not None
