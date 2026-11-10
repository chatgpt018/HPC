from pathlib import Path
import pytest
from marketintel.documents import ingest_file, parse_file


def test_real_docx(monkeypatch,tmp_path):
    from docx import Document
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'store'))
    d=Document()
    d.add_paragraph('Solar capacity and module procurement in this example document.')
    d.add_table(rows=1,cols=2)
    d.tables[0].cell(0,0).text='Sector'
    d.tables[0].cell(0,1).text='Renewable energy'
    path=tmp_path/'example.docx';d.save(path)
    out=ingest_file(path)
    assert out['new_chunks']>=2
    assert 'paragraph 1' in [x[0] for x in parse_file(path)]


def test_real_pdf(monkeypatch,tmp_path):
    import fitz
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'store'))
    pdf=fitz.open();page=pdf.new_page();page.insert_text((72,72),'Synthetic market document for parsing validation.')
    path=tmp_path/'report.pdf';pdf.save(path);pdf.close()
    result=ingest_file(path)
    assert result['new_chunks']==1
    assert parse_file(path)[0][0]=='page 1'


def test_bad_pdf_has_error(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'store'))
    path=tmp_path/'corrupt.pdf';path.write_bytes(b'not a PDF')
    with pytest.raises(Exception):
        ingest_file(path)
