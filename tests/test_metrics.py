import pytest
from marketintel.documents import ingest_file
from marketintel.metrics import growth_table


def test_real_csv_metrics(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'data'))
    csv=tmp_path/'valid.csv'
    csv.write_text('market,year,value,unit,region,currency\nSolar,2023,100,billion,India,INR\nSolar,2024,110,billion,India,INR\n',encoding='utf-8')
    ingest_file(csv)
    result,warnings=growth_table()
    assert len(result)==1 and not warnings
    assert result[0]['yoy_percent']==pytest.approx(10)
    assert result[0]['series_cagr_percent']==pytest.approx(10)
    assert 'valid.csv row 3'==result[0]['source']


def test_conflicting_duplicate_omitted(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'data'))
    csv=tmp_path/'dupes.csv'
    csv.write_text('market,year,value,unit,region,currency\nSolar,2023,100,billion,India,INR\nSolar,2023,125,billion,India,INR\nSolar,2024,110,billion,India,INR\n')
    ingest_file(csv)
    result,warnings=growth_table()
    assert result==[]
    assert any('duplicate' in warning for warning in warnings)
