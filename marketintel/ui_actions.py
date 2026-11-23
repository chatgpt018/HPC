"""UI actions backed by real project data, never simulated MPI metrics."""
from __future__ import annotations

import json
import csv
import math
import re
from html import escape
import uuid
import statistics
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import sequential_kmeans, sequential_similarity
from .documents import ingest_file
from .jobs import mpi_available, run_local_mpi, saved_jobs
from .metrics import growth_table
from .provenance import attach_provenance, research_brief
from .report_generation import explain
from .rag import answer
from .storage import chunks, home, records, write_json
from .vectors import load_dataset, save_dataset


def overview() -> tuple[str, str]:
    """Return actual local measurements as KPI markup and honest mode information."""
    from collections import Counter
    from html import escape
    docs = records()
    report_chunks = chunks()
    jobs = saved_jobs()
    result_count = len(_latest_results())
    validated = sum(1 for j in jobs if j.get('status') == 'COMPLETED')
    metrics = [
        ('Documents', len(docs), 'Uploaded sources'),
        ('Source excerpts', len(report_chunks), 'Traceable passages'),
        ('Saved analyses', result_count, 'Computed output files'),
        ('MPI jobs', validated, 'Verified completed attempts'),
    ]
    cards = ''.join(
        f'<article class="mi-kpi"><span class="mi-kpi-label">{escape(name)}</span>'
        f'<strong>{number:,}</strong><small>{escape(sub)}</small></article>'
        for name, number, sub in metrics
    )
    categories = Counter(d.get('industry', 'Unspecified') for d in docs)
    category_note = ', '.join(f'{escape(str(name))} ({count})' for name, count in categories.most_common(4))
    if not category_note:
        category_note = 'No documents indexed yet — open Documents & AI to begin.'
    summary = f'<div class="mi-kpis">{cards}</div><div class="mi-data-note"><strong>Indexed collection:</strong> {category_note}</div>'
    mode = 'Launcher detected — real execution still requires successful MPI validation' if mpi_available() else 'MPI launcher not detected; use WSL2 / Ubuntu for MPI'
    environment = (f'**Execution environment:** Local-only; AWS is **not deployed**.  '
                   f'\n\n**MPI:** {mode}.  '
                   '**Retrieval:** local TF-IDF by default; neural mode requires a locally configured model.  '
                   '**Text generation:** evidence-only unless you explicitly enable a local Ollama model.')
    return summary, environment


def upload(files, industry: str):
    if not files:
        return 'Choose at least one PDF, DOCX or CSV file.', table_documents()
    feedback=[]
    for file in files:
        try:
            out=ingest_file(Path(file),industry)
            feedback.append(f"✓ {out['document']['filename']}: {out['status']} ({out['new_chunks']} new chunks)")
        except Exception as exc:
            feedback.append(f'✕ {Path(file).name}: {exc}')
    return '\n'.join(feedback),table_documents()


def table_documents():
    return [[d['filename'],d['industry'],d['chunk_count']]
            for d in records()]


def document_options():
    """Select the newest indexed document without exposing local file paths."""
    documents = records()
    return gr.update(
        choices=[(d['filename'], d['doc_id']) for d in documents],
        value=documents[-1]['doc_id'] if documents else None,
    )


def document_overview(doc_id: str | None = None) -> str:
    """Show literal source excerpts and terms, never an invented PDF summary."""
    from collections import Counter
    from html import escape
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    documents = records()
    if not documents:
        return '<div class="mi-doc-overview">Index a PDF, Word or CSV file to preview its extracted text here.</div>'
    selected = next((d for d in documents if d['doc_id'] == doc_id), None) if doc_id else documents[-1]
    if selected is None:
        return '<div class="mi-doc-overview">Choose an indexed document to inspect its text.</div>'
    passages = [c for c in chunks() if c['doc_id'] == selected['doc_id']]
    if not passages:
        return '<div class="mi-doc-overview">No readable passages are indexed for this document.</div>'

    source_count = len({c['source'] for c in passages})
    is_pdf = all(c['source'].startswith('page ') for c in passages)
    location_label = ('page' if source_count == 1 else 'pages') if is_pdf else (
        'source location' if source_count == 1 else 'source locations'
    )
    common = Counter(
        word for word in re.findall(r'[a-z][a-z0-9-]{2,}', ' '.join(c['text'] for c in passages).lower())
        if word not in ENGLISH_STOP_WORDS and word not in {
            'document', 'report', 'reports', 'page', 'slide', 'slides',
            'exam', 'line', 'map', 'data', 'answer', 'question', 'section', 'topic',
            'summary', 'synthetic',
        }
    )
    terms = [word for word, _ in common.most_common(4)]
    previews = []
    seen_sources = set()
    for passage in passages:
        if passage['source'] in seen_sources:
            continue
        seen_sources.add(passage['source'])
        text = passage['text'].strip()
        snippet = escape(text[:300].rsplit(' ', 1)[0] + '…' if len(text) > 300 else text)
        previews.append(f'<li><strong>{escape(passage["source"])}</strong><p>{snippet}</p></li>')
        if len(previews) == 3:
            break
    terms_html = ', '.join(escape(term) for term in terms) if terms else 'No repeated topic words found'
    attribution = ' · '.join(
        f'<strong>{escape(label)}:</strong> {escape(str(selected[key]))}'
        for key, label in (('source_kind', 'Source status'), ('publisher', 'Publisher'),
                           ('publication_date', 'Published'), ('geographic_scope', 'Scope'),
                           ('source_url', 'Source URL'))
        if selected.get(key)
    )
    questions = ''.join(
        f'<li>What does this document say about {escape(term)}?</li>'
        for term in terms[::2][:2]
    )
    return (
        '<div class="mi-doc-overview">'
        f'<h4>{escape(selected["filename"])}</h4>'
        f'<p><strong>Indexed:</strong> {len(passages)} text passages from {source_count} {location_label}. '
        'This is an extractive preview, not an AI-written summary.</p>'
        + (f'<p>{attribution}</p>' if attribution else '')
        + f'<p><strong>Frequent terms:</strong> {terms_html}</p>'
        '<h5>What the source text says</h5><ul class="mi-source-preview">'
        + ''.join(previews) + '</ul>'
        + ('<h5>Questions to try</h5><ul class="mi-questions">' + questions + '</ul>' if questions else '')
        + '</div>'
    )


def analysis_guidance(operation: str, count: int, mode: str):
    """Explain what the selected analysis will produce before it can run."""
    total = len(records())
    if total == 0:
        note = '**No documents indexed.** Add at least two documents in Documents & AI first.'
        ready = False
    elif total == 1:
        note = '**1 document indexed.** You can read and search it now. Add another document to compare or cluster them.'
        ready = False
    elif operation == 'K-Means clustering' and count > total:
        note = f'**{total} documents indexed.** Choose {total} or fewer groups, or add more documents.'
        ready = False
    elif operation == 'K-Means clustering':
        note = f'**Ready:** {total} documents will be assigned to {count} groups. The output will list each document and its group.'
        ready = True
    else:
        note = f'**Ready:** {total} documents will be compared. The output will show matching documents and cosine scores.'
        ready = True
    if ready:
        note += (' Real MPI will start separate ranks only if your MPI runtime works; there is no simulated fallback.'
                 if mode == 'Real MPI' else ' Sequential mode runs locally without MPI.')
    return note, gr.update(interactive=ready)


def ask(question: str, use_ollama: bool):
    try:
        res = answer(question, generate=use_ollama)
        refs = [[p['filename'],p['source'],round(p['score'],3),p['text'][:350]] for p in res['evidence']]
        if res['mode'] == 'ollama-abstained':
            label = f'Local Ollama model {res["model"]} consulted; evidence insufficient'
        elif res['generated']:
            label = f'Real local Ollama generation · model {res["model"]}'
        else:
            label = 'Evidence-only, no LLM invocation'
        retrieval_name = {'tfidf': 'TF-IDF', 'semantic': 'Semantic embeddings'}.get(
            res.get('retrieval_mode'), 'Unknown')
        retrieval = f'**Retrieval:** {retrieval_name}'
        if not refs:
            help_text = ('Read the extracted document preview above and try one of its topic-specific questions.'
                         if records() else 'Index a document first, then ask about its contents.')
            return f'**Mode:** {label} · {retrieval}\n\n**No matching passage.** {help_text}', []
        review = ('\n\n**Citation check:** Source numbers refer to retrieved passages; '
                  'check each generated claim against those passages before using it as a finding.'
                  if res['generated'] else '')
        return f"**Mode:** {label} · {retrieval}\n\n{res['answer']}{review}",refs
    except Exception as exc:
        return f'**Could not answer:** {exc}',[]


def _latest_results():
    return sorted((home()/'results').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)


def run_analysis(operation: str, mode: str, processes: int, count: int, embedding_mode: str):
    try:
        if operation not in {'Exact similarity · top-k', 'K-Means clustering'}:
            raise ValueError('Choose a supported analysis method')
        if mode not in {'Sequential', 'Real MPI'}:
            raise ValueError('Choose Sequential or Real MPI execution')
        dataset_path=home()/'datasets'/'uploaded_reports.npz'
        detail=save_dataset(dataset_path, embedding_mode=embedding_mode)
        ids,matrix=load_dataset(dataset_path)
        if operation == 'Exact similarity · top-k' and len(ids) < 2:
            raise ValueError('Similarity needs at least two indexed documents. Add another PDF, Word, or CSV report and run again.')
        if operation == 'K-Means clustering' and len(ids) < 2:
            raise ValueError('Clustering needs at least two indexed documents. Add another report and run again.')
        if operation == 'K-Means clustering' and count>len(ids):
            raise ValueError(f'Cluster count {count} exceeds {len(ids)} uploaded documents')
        if mode == 'Real MPI':
            entry=run_local_mpi('clustering' if operation == 'K-Means clustering' else 'similarity',
                                processes,k=count,top_k=count,embedding_mode=embedding_mode,
                                dataset_path=dataset_path)
            result=json.loads(Path(entry['result_path']).read_text(encoding='utf-8'))
            mode_text=f"**REAL MPI execution:** job `{entry['run_id']}`, {processes} requested processes; actual ranks recorded."
        else:
            result=(sequential_kmeans(ids,matrix,k=count) if operation=='K-Means clustering'
                    else sequential_similarity(ids,matrix,k=count))
            run_id=uuid.uuid4().hex[:12]
            result=attach_provenance(result,ids,matrix,detail['vector_type'])
            dest=home()/'results'/f'{run_id}.json'
            write_json(dest,result)
            dest.with_suffix('.md').write_text(research_brief(result), encoding='utf-8')
            mode_text=f'**Sequential local computation** · run `{run_id}` · MPI not used.'
        meta=result['metadata']
        details=f"{mode_text}\n\n**Input:** {detail['documents']} uploaded documents; "\
                f"{detail['vector_type']} features. **Measured runtime:** {meta['runtime_seconds']:.5f} s."
        names={d['doc_id']:d['filename'] for d in records()}
        rows = ([[names.get(x['doc_id'],x['doc_id']),names.get(x['similar_doc_id'],x['similar_doc_id']),round(x['score'],5)] for x in result['matches'][:70]]
                if 'matches' in result else
                [[names.get(x['doc_id'],x['doc_id']),str(x['cluster_id']),''] for x in result['assignments'][:70]])
        details += f"\n\n**Dataset SHA-256:** `{meta.get('input_sha256', 'not recorded')[:16]}…` · **Source traceability:** {len(meta.get('source_documents',[]))} documents."
        return details,rows,json.dumps(meta,indent=2)
    except Exception as exc:
        return f'**Analysis not completed:** {exc}',[],''


def list_results():
    names=[p.name for p in _latest_results()]
    return gr.update(choices=names, value=names[0] if names else None)


def inspect_result(filename):
    if not filename:
        return 'No saved analysis yet.',[],''
    # Never accept an arbitrary path from a UI dropdown.
    path=(home()/'results'/Path(filename).name)
    if not path.is_file():
        return 'Result no longer exists.',[],''
    data=json.loads(path.read_text(encoding='utf-8'))
    meta=data.get('metadata',{})
    entries=data.get('matches',data.get('assignments',[]))
    names={d['doc_id']:d['filename'] for d in records()}
    rows=([[names.get(r['doc_id'],r['doc_id']),names.get(r['similar_doc_id'],r['similar_doc_id']),str(round(r['score'],5))]
           if 'score' in r else [names.get(r['doc_id'],r['doc_id']),f"Cluster {r['cluster_id']}",'']
           for r in entries[:200]])
    elapsed = meta.get('runtime_seconds')
    elapsed_label = f'{elapsed:.3g}' if isinstance(elapsed, (int, float)) else 'not recorded'
    return (f"**{meta.get('mode','Unknown')}** · {meta.get('algorithm','')} · "
            f"{meta.get('documents','?')} documents · runtime {elapsed_label} seconds",
            rows,json.dumps(meta,indent=2))



def result_chart(filename):
    if not filename:
        return None
    path=home()/'results'/Path(filename).name
    if not path.is_file():
        return None
    data=json.loads(path.read_text(encoding='utf-8'))
    fig=go.Figure()
    if 'assignments' in data:
        from collections import Counter
        groups=Counter(r['cluster_id'] for r in data['assignments'])
        fig.add_trace(go.Bar(x=[f'Cluster {x}' for x in sorted(groups)],y=[groups[x] for x in sorted(groups)]))
        fig.update_layout(title='Documents per computed cluster', yaxis_title='Documents')
    elif 'matches' in data:
        doc_names={d['doc_id']:d['filename'] for d in records()}
        items=sorted(data['matches'],key=lambda a:-a['score'])[:10]
        fig.add_trace(go.Bar(x=[r['score'] for r in items],y=[doc_names.get(r['doc_id'],r['doc_id'])[:35] for r in items],orientation='h'))
        fig.update_layout(title='Top computed document similarities',xaxis_title='Cosine similarity')
    else:
        return None
    fig.update_layout(template='plotly_white',height=340,margin=dict(l=30,r=20,t=55,b=40))
    return fig


def benchmark_options():
    choices=[(label,key) for key,(_,_,label) in _benchmark_files().items()
             if key.startswith(('bundled:','local:'))]
    return gr.update(choices=choices,value=choices[0][1] if choices else None)


def _benchmark_files():
    """Resolve only files discovered beneath trusted benchmark roots."""
    found={}
    bundled=Path(__file__).resolve().parent.parent/'output'/'benchmarks'
    local=home()/'benchmarks'
    for root,prefix in ((bundled,'bundled'),(local,'local')):
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            study=folder/'scaling_study.json'
            if not study.is_file():
                study=folder/'scaling_study.csv'
            if study.is_file() and study.resolve().is_relative_to(root.resolve()):
                key=f'{prefix}:study:{folder.name}'
                label=(f'Historical Windows MPI study · {folder.name}' if prefix=='bundled'
                       else f'Local MPI study · {folder.name}')
                found[key]=(study,'study',label)
        if prefix=='local':
            for path in sorted(root.rglob('*_benchmark_*p.json')):
                if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                    name=path.relative_to(root).as_posix()
                    found[f'local:run:{name}']=(path,'run',f'Local MPI run · {name}')
                    if '/' not in name:
                        found[name]=(path,'run',f'Local MPI run · {name}')
    return found


def load_benchmarks():
    options=benchmark_options()
    summary,plot=show_benchmark(options['value'])
    return options,summary,plot


def _positive_number(value, name):
    number=float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'Invalid {name}')
    return number


def _positive_int(value, name):
    number=_positive_number(value,name)
    if not number.is_integer():
        raise ValueError(f'Invalid {name}')
    return int(number)


def _study_rows(path):
    if path.suffix=='.csv':
        with path.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        metadata={}
    else:
        record=json.loads(path.read_text(encoding='utf-8'))
        rows=record['rows']
        metadata=record
    if not isinstance(rows,list) or not 0 < len(rows) <= 1000:
        raise ValueError('Study has no valid rows')
    cleaned=[]
    for row in rows:
        if row['operation'] not in {'similarity','clustering'}:
            raise ValueError('Invalid algorithm')
        verified=row['verified'] if isinstance(row['verified'],bool) else row['verified']=='True'
        item={
            'operation':row['operation'],
            'documents':_positive_int(row['documents'],'dataset size'),
            'dimensions':_positive_int(row['dimensions'],'dimension'),
            'mpi_processes':_positive_int(row['mpi_processes'],'rank count'),
            'repetitions':_positive_int(row['repetitions'],'repetitions'),
            'seq':_positive_number(row['median_sequential_seconds'],'sequential time'),
            'mpi':_positive_number(row['median_mpi_seconds'],'MPI time'),
            'speedup':_positive_number(row['speedup'],'speedup'),
            'efficiency':_positive_number(row['efficiency_percent'],'efficiency'),
            'seq_throughput':_positive_number(row['sequential_work_units_per_second'],'sequential throughput'),
            'mpi_throughput':_positive_number(row['mpi_work_units_per_second'],'MPI throughput'),
            'hosts':str(row.get('hosts_observed','')).strip(),
            'verified':verified,
        }
        if (not math.isclose(item['speedup'],item['seq']/item['mpi'],rel_tol=1e-5) or
                not math.isclose(item['efficiency'],100*item['speedup']/item['mpi_processes'],rel_tol=1e-5)):
            raise ValueError('Inconsistent performance metrics')
        cleaned.append(item)
    return cleaned,metadata


def _study_chart(rows):
    fig=make_subplots(rows=2,cols=2,subplot_titles=('End-to-end runtime · seconds',
        'Speedup · Sequential / MPI','Parallel efficiency · %','MPI throughput · work units/s'))
    groups=sorted({(r['operation'],r['mpi_processes']) for r in rows})
    for operation,processes in groups:
        series=sorted((r for r in rows if (r['operation'],r['mpi_processes'])==(operation,processes)),
                      key=lambda r:r['documents'])
        x=[r['documents'] for r in series]
        label=f'{operation} · {processes} ranks'
        for field in ('seq','mpi'):
            fig.add_trace(go.Scatter(x=x,y=[r[field] for r in series],mode='lines+markers',
                name=f'{label} · {field}',legendgroup=f'{label}-{field}',showlegend=True,
                line=dict(dash='dash' if field=='seq' else 'solid')),row=1,col=1)
        for row,col,field in ((1,2,'speedup'),(2,1,'efficiency'),(2,2,'mpi_throughput')):
            fig.add_trace(go.Scatter(x=x,y=[r[field] for r in series],mode='lines+markers',
                name=label,legendgroup=label,showlegend=False),row=row,col=col)
    fig.update_xaxes(title_text='Documents')
    fig.update_layout(template='plotly_white',height=750,
                      title='Recorded scaling measurements',
                      margin=dict(l=45,r=25,t=85,b=45))
    return fig


def show_benchmark(filename):
    if not filename:
        return 'No real MPI benchmark files found. Bundled studies and new local runs appear here after refresh.',None
    selected=_benchmark_files().get(filename)
    if selected is None:
        return 'Benchmark selection not found. Refresh the list and select a saved measurement.',None
    path,kind,label=selected
    try:
        if path.stat().st_size > 5_000_000:
            raise ValueError('Benchmark file is too large')
        if kind=='study':
            rows,_=_study_rows(path)
            historical=filename.startswith('bundled:')
            source=('**Historical Windows MPI measurements**' if historical
                    else '**New local MPI study**')
            hosts={host.strip() for r in rows for host in re.split(r'[,;]',r['hosts']) if host.strip()}
            host_note=(f'**Single host:** {escape(next(iter(hosts)))}' if len(hosts)==1
                       else f'**Recorded hosts:** {len(hosts)}' if hosts else '**Hosts:** not recorded')
            header=(f'{source} · **Input:** synthetic numerical benchmark · {host_note} · '
                    f'**Configurations:** {len(rows)} · **Correctness:** '
                    f'{sum(r["verified"] for r in rows)}/{len(rows)} verified. '
                    'End-to-end time includes MPI launch and output; speedup below 1 means MPI was slower. '
                    'Throughput is measured work units per second, not market growth.')
            table=['| Algorithm | Docs | Dim | Ranks | Reps | Seq s | MPI s | Speedup | Efficiency | Seq/s | MPI/s | Correct |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
            for r in rows:
                table.append(f'| {r["operation"]} | {r["documents"]:,} | {r["dimensions"]} | '
                    f'{r["mpi_processes"]} | {r["repetitions"]} | {r["seq"]:.3f} | '
                    f'{r["mpi"]:.3f} | {r["speedup"]:.3f}× | {r["efficiency"]:.2f}% | '
                    f'{r["seq_throughput"]:,.0f} | {r["mpi_throughput"]:,.0f} | '
                    f'{"Verified" if r["verified"] else "Failed"} |')
            return header+'\n\n'+'\n'.join(table),_study_chart(rows)
        record=json.loads(path.read_text(encoding='utf-8'))
        return _show_run_benchmark(record)
    except (OSError,UnicodeError,ValueError,TypeError,KeyError,json.JSONDecodeError,OverflowError):
        return 'Saved benchmark is malformed or incomplete. It was not displayed as a verified result.',None


def _show_run_benchmark(record):
    runs=record['runs']
    if not isinstance(runs,list) or not 0<len(runs)<=1000:
        raise ValueError('Invalid repetitions')
    p=_positive_int(record['mpi_processes'],'rank count')
    seq=[_positive_number(r['sequential_wall_seconds'],'sequential time') for r in runs]
    mpi=[_positive_number(r['mpi_end_to_end_wall_seconds'],'MPI time') for r in runs]
    fig=go.Figure()
    fig.add_trace(go.Bar(name='Sequential',x=[f'Run {i+1}' for i in range(len(runs))],y=seq))
    fig.add_trace(go.Bar(name=f'Real MPI ({p} processes)',x=[f'Run {i+1}' for i in range(len(runs))],y=mpi))
    fig.update_layout(barmode='group',template='plotly_white',height=340,yaxis_title='End-to-end seconds',
                      title='Actual measured runtime (includes process startup)')
    speed=_positive_number(record.get('median_speedup',statistics.median(seq)/statistics.median(mpi)),'speedup')
    throughput=record.get('median_mpi_work_units_per_second')
    perf=(f' · **Throughput:** {_positive_number(throughput,"throughput"):,.0f} work units/s'
          if throughput is not None else ' · **Throughput:** not recorded')
    kind=escape(str(record.get('dataset_kind','Input provenance not recorded — inspect the source dataset')))
    operation=record['operation']
    if operation not in {'similarity','clustering'}:
        raise ValueError('Invalid algorithm')
    correctness=('Verified' if all(r.get('correct') is True for r in runs) else
                 'Failed' if any(r.get('correct') is False for r in runs) else 'Not recorded')
    dimensions=_positive_int(record['dimensions'],'dimension') if 'dimensions' in record else 'not recorded'
    documents=_positive_int(record['documents'],'dataset size')
    return (f'**New local MPI run** · **Input:** {kind} · **Algorithm:** {operation} · '
            f'**Documents:** {documents} · **Dimension:** {dimensions} · '
            f'**Ranks:** {p} · **Repetitions:** {len(runs)} · '
            f'**Sequential:** {statistics.median(seq):.3f} s · **MPI:** {statistics.median(mpi):.3f} s · '
            f'**Speedup:** {speed:.3f}× · **Efficiency:** {100*speed/p:.2f}%'
            +perf+f' · **Correctness:** {correctness}. '
            'End-to-end MPI time includes startup; speedup below 1 is a slowdown.'),fig


def display_market_metrics():
    computed,warnings=growth_table()
    if not computed:
        message='No comparable time-series rows found. Upload a CSV with market,year,value,unit,region,currency columns and at least two years per series.'
        if warnings:
            message+=' Warnings: '+ '; '.join(warnings[:3])
        return message,[],None
    table=[[r['market'],r['region'],r['year'],round(r['value'],3),r['unit'],
            r['currency'],'' if r['yoy_percent'] is None else round(r['yoy_percent'],2),
            round(r['series_cagr_percent'],2) if r['series_cagr_percent'] is not None else '',r['source']]
           for r in computed]
    fig=go.Figure()
    by_market={}
    for r in computed:
        by_market.setdefault((r['market'],r['region'],r['unit'],r['currency']),[]).append(r)
    for (market,region,unit,currency),points in list(by_market.items())[:6]:
        pts=sorted(points,key=lambda x:x['year'])
        fig.add_trace(go.Scatter(name=f'{market} · {region} ({unit}, {currency})',
                                 x=[p['year'] for p in pts],y=[p['value'] for p in pts],mode='lines+markers'))
    fig.update_layout(title='Validated market time-series samples (not forecasts)',
                      template='plotly_white',height=350,yaxis_title='Source-reported value')
    return (f'**{len(by_market)}** comparable series analysed. Values must be checked against the original source. '
            +('; '.join(warnings[:3]) if warnings else 'No parsing warnings.')),table,fig


def explain_saved_result(filename, use_ollama=False):
    if not filename:
        return 'Select a saved result first.'
    file=home()/'results'/Path(filename).name
    if not file.is_file():
        return 'Selected result is not available.'
    try:
        output=explain(json.loads(file.read_text(encoding='utf-8')), use_ollama=use_ollama)
        if output['generated']:
            return (f"**Local Ollama-generated explanation (model: {output['model']})**\n\n"
                    f"{output['text']}\n\n**Retrieved sources**\n{output['references']}")
        return '**Deterministic computational brief (no LLM invoked)**\n\n'+output['text']
    except Exception as exc:
        return f'**Explanation unavailable:** {exc}'

def download_brief(filename):
    if not filename:
        raise gr.Error('Select an actual result first')
    path=home()/'results'/Path(filename).name
    if not path.is_file():
        raise gr.Error('Result file was not found')
    result=json.loads(path.read_text(encoding='utf-8'))
    brief=path.with_suffix('.md')
    brief.write_text(research_brief(result),encoding='utf-8')
    return str(brief)

def download_result(filename):
    if not filename:
        raise gr.Error('Select a saved result first')
    path=home()/'results'/Path(filename).name
    if not path.is_file():
        raise gr.Error('The selected result file was not found')
    return str(path)



def workspace_quality():
    """Display observed data coverage, not a fabricated trust/quality score."""
    from .research_quality import quality_report
    from collections import Counter
    q = quality_report()
    note = '\n'.join(f'- {item}' for item in q['observations']) or '- No obvious structural warnings; verify factual claims with original sources.'
    summary = (f"**{q['documents']} indexed files** · **{q['passages']} source excerpts** · "
        f"**{q['expected_directed_pairs']:,} directed document comparisons**\n\n"
        f"A dense {q['documents']} × {q['documents']} float32 similarity matrix alone would use "
        f"approximately **{q['estimated_dense_matrix_mib']:,.2f} MiB**. The implementation computes top-k in blocks. "
        'Document count is not proof of an HPC speedup.\n\n'
        f'**Data-readiness notes**\n\n{note}\n\n'
        '*This audit checks evidence coverage and computational size, not the accuracy of the reports.*')
    chart = go.Figure()
    counts = q['industries']
    if counts:
        chart.add_trace(go.Bar(y=list(counts), x=list(counts.values()), orientation='h',
                               marker_color='#2a8076', name='Indexed files'))
        chart.update_layout(title='Source coverage by provided industry',
                            xaxis_title='Indexed documents', yaxis_title='Industry')
    else:
        chart.add_annotation(text='No documents indexed yet', x=.5, y=.5,
                             xref='paper', yref='paper', showarrow=False)
        chart.update_xaxes(visible=False)
        chart.update_yaxes(visible=False)
    chart.update_layout(template='plotly_white', height=270,
                        margin=dict(l=20,r=20,t=45,b=20), showlegend=False,
                        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff')
    return summary, chart


def import_practice_collection():
    """Explicitly import only the bundled SYNTHETIC teaching files."""
    folder = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_reports'
    if not folder.is_dir():
        return 'Synthetic practice examples are missing.', table_documents()
    results=[]
    for source in sorted(folder.glob('synthetic_*.csv')):
        try:
            label=source.stem.split('_',2)[-1].replace('_',' ').title()
            out=ingest_file(source, industry='SYNTHETIC / '+label,
                            source_metadata={'source_kind': 'synthetic'})
            results.append(out['status'])
        except Exception as exc:
            results.append(f'ERROR {source.name}: {exc}')
    return (f'**SYNTHETIC DEMO ONLY:** {len(results)} practice files processed. '
            'These are fictional examples, not market evidence. ' +
            ', '.join(results[:5])), table_documents()


def compute_preview():
    """Compute the exact pair count from user-uploaded document metadata."""
    from .research_quality import quality_report
    data = quality_report()
    n = data['documents']
    if n < 2:
        return '**No HPC task available:** upload at least two documents, then calculate similarity or clustering.'
    msg = f'**Actual current workload:** {n:,} documents → {data["expected_directed_pairs"]:,} directed non-self similarities.'
    if n < 100:
        msg += '\n\n**HPC justification:** this is a small collection. Sequential execution is usually preferable; use larger validated data to study speedup.'
    else:
        msg += '\n\n**HPC justification:** quadratic all-pairs computation creates a scaling experiment. Verify the benefit with real MPI benchmarks.'
    return msg
