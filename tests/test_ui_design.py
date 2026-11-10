"""Public-facing UI contracts; these checks require no cloud or MPI service."""
from pathlib import Path
from html import unescape

from marketintel.storage import write_json
from marketintel.ui import build
from marketintel.ui_actions import overview
from marketintel.ui_theme import TITLE, CSS, MASTHEAD


def test_full_submitted_title_remains_unchanged():
    assert TITLE == 'AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS'
    assert TITLE in unescape(MASTHEAD)


def test_exactly_four_clear_destinations():
    config = build().get_config_file()
    tabs = [component['props']['label'] for component in config['components']
            if component['type'] == 'tabitem']
    assert tabs == [
        '01  Overview', '02  Documents & AI', '03  HPC Analytics',
        '04  Results & Reports',
    ]


def test_dashboard_has_consistent_local_style():
    assert 'mi-masthead' in CSS
    assert 'mi-kpis' in CSS
    assert 'mi-workflow' in CSS
    assert '@media(max-width:800px)' in CSS


def test_overview_counts_real_local_records(tmp_path: Path, monkeypatch):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path))
    write_json(tmp_path / 'documents.json', [
        {'filename': 'one.pdf', 'industry': 'Energy'},
        {'filename': 'two.pdf', 'industry': 'Energy'},
    ])
    write_json(tmp_path / 'chunks.json', [{'text': 'one'}, {'text': 'two'}, {'text': 'three'}])
    write_json(tmp_path / 'jobs' / 'test.json', {'status': 'FAILED', 'started_utc': '2026-11-09'})
    write_json(tmp_path / 'results' / 'actual.json', {'metadata': {'mode': 'sequential'}})
    summary, environment = overview()
    assert '2</strong>' in summary  # Indexed document count
    assert '3</strong>' in summary  # Traceable passage count
    assert '1</strong>' in summary  # Saved analysis count
    assert 'Energy (2)' in summary
    assert 'AWS is **not deployed**' in environment
    assert 'MPI' in environment
