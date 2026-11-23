"""Local interface-action checks; no browser, AWS, MPI runtime or LLM required."""
import json
from pathlib import Path

from marketintel.ui_actions import (
    upload, ask, run_analysis, inspect_result, result_chart, show_benchmark,
    benchmark_options, load_benchmarks,
    document_options, document_overview, analysis_guidance,
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
    status, table, metadata = run_analysis('K-Means clustering', 'Sequential', 2, 1, 'hashing')
    assert 'at least two indexed documents' in status
    assert table == [] and metadata == ''
    assert not list((home() / 'results').glob('*.json'))


def test_document_overview_is_cited_and_escapes_uploaded_text(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    write_json(home() / 'documents.json', [
        {'doc_id': 'one', 'filename': 'NISQ <notes>.pdf', 'industry': 'Synthetic', 'chunk_count': 2},
    ])
    write_json(home() / 'chunks.json', [
        {'chunk_id': 'one-00000', 'doc_id': 'one', 'source': 'page 1',
         'text': 'Quantum circuits and NISQ hardware. <script>alert(1)</script>'},
        {'chunk_id': 'one-00001', 'doc_id': 'one', 'source': 'page 2',
         'text': 'Quantum circuits need error correction and reliable qubits.'},
    ])
    preview = document_overview()
    assert 'page 1' in preview and 'page 2' in preview
    assert 'Questions to try' in preview and 'Frequent terms' in preview
    assert '&lt;notes&gt;' in preview and '&lt;script&gt;' in preview
    assert '<script>' not in preview
    assert document_options()['value'] == 'one'

    response, evidence = ask('what is this pdf about', False)
    assert 'document preview' in response and evidence == []


def test_analysis_guidance_prevents_invalid_one_document_job(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    write_json(home() / 'documents.json', [{'doc_id': 'one', 'filename': 'one.pdf'}])
    note, button = analysis_guidance('K-Means clustering', 3, 'Sequential')
    assert '1 document indexed' in note and button['interactive'] is False

    write_json(home() / 'documents.json', [
        {'doc_id': 'one', 'filename': 'one.pdf'}, {'doc_id': 'two', 'filename': 'two.pdf'},
    ])
    note, button = analysis_guidance('K-Means clustering', 3, 'Sequential')
    assert '2 or fewer groups' in note and button['interactive'] is False
    note, button = analysis_guidance('K-Means clustering', 2, 'Sequential')
    assert 'Ready' in note and button['interactive'] is True


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


def test_bundled_windows_study_is_visible_without_local_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'empty-store'))
    options, summary, chart = load_benchmarks()
    assert options['value'].startswith('bundled:study:')
    assert len(options['choices']) >= 2
    assert 'Historical Windows MPI measurements' in summary
    assert 'Single host' in summary and '12/12 verified' in summary
    assert '| similarity | 100 | 384 | 2 | 3 |' in summary
    assert '0.009×' in summary and '0.44%' in summary
    assert chart is not None and len(chart.data) == 20


def test_larger_bundled_study_displays_measured_scaling(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'empty-store'))
    summary, chart = show_benchmark('bundled:study:native_windows_20261122_large')
    assert 'Historical Windows MPI measurements' in summary
    assert '8/8 verified' in summary and 'Single host' in summary
    assert '| similarity | 5,000 | 384 | 4 | 3 |' in summary
    assert '0.350×' in summary
    assert chart is not None and len(chart.data) == 20


def test_new_local_study_and_csv_fallback_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    bundled = Path(__file__).resolve().parents[1] / 'output' / 'benchmarks' / 'native_windows_20261122'
    study = home() / 'benchmarks' / 'fresh-study'
    study.mkdir(parents=True)
    (study / 'scaling_study.csv').write_bytes((bundled / 'scaling_study.csv').read_bytes())
    options = benchmark_options()
    keys = [value for _, value in options['choices']]
    assert any(key.startswith('bundled:study:') for key in keys)
    assert 'local:study:fresh-study' in keys
    summary, chart = show_benchmark('local:study:fresh-study')
    assert 'New local MPI study' in summary and '12/12 verified' in summary
    assert chart is not None


def test_malformed_or_unlisted_benchmark_cannot_be_displayed(tmp_path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'local-store'))
    study = home() / 'benchmarks' / 'broken'
    study.mkdir(parents=True)
    (study / 'scaling_study.json').write_text('{bad json', encoding='utf-8')
    message, chart = show_benchmark('local:study:broken')
    assert 'malformed' in message and chart is None
    message, chart = show_benchmark('../output/benchmarks/native_windows_20261122/scaling_study.json')
    assert 'selection not found' in message and chart is None
