import json
from marketintel.documents import ingest_file
from marketintel.provenance import attach_provenance
from marketintel.report_generation import explain
from marketintel.vectors import save_dataset,load_dataset
from marketintel.analysis import sequential_similarity


def test_combined_report_uses_real_computation_and_source_excerpts(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'state'))
    for name,text in [('ev.csv','electric battery EV growth discussion'),('solar.csv','solar photovoltaic renewable capacity')]:
        file=tmp_path/name
        file.write_text('subject,details\nmarket,'+text+'\n',encoding='utf-8')
        ingest_file(file)
    manifest=save_dataset(tmp_path/'matrix.npz')
    ids,data=load_dataset(tmp_path/'matrix.npz')
    result=attach_provenance(sequential_similarity(ids,data,k=1),ids,data,manifest['vector_type'])
    basic=explain(result,use_ollama=False)
    assert basic['generated'] is False
    assert 'source traceability' in basic['text'].lower()
    requests=[]
    class Response:
        def __enter__(self): return self
        def __exit__(self,*exc): return False
        def read(self): return json.dumps({'response':'The similarity calculation grouped the reports by words.'}).encode()
    def fake_call(req, timeout):
        requests.append(json.loads(req.data))
        return Response()
    monkeypatch.setenv('MARKETINTEL_OLLAMA_MODEL','local-test-model')
    monkeypatch.setattr('marketintel.local_llm.urllib.request.urlopen',fake_call)
    out=explain(result,use_ollama=True)
    assert out['generated'] and out['sources']
    assert 'verified computational summary' in requests[0]['prompt'].lower()
    assert 'ev.csv' in requests[0]['prompt']
    assert result['metadata']['input_sha256'] in requests[0]['prompt']
