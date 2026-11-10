"""UI actions backed by real project data, never simulated MPI metrics."""
from __future__ import annotations

import json
import uuid
import statistics
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

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
    return [[d['filename'],d['industry'],d['chunk_count'],d['doc_id']]
            for d in records()]


def ask(question: str, use_ollama: bool):
    try:
        res = answer(question, generate=use_ollama)
        refs = [[p['filename'],p['source'],round(p['score'],3),p['text'][:350]] for p in res['evidence']]
        label = ('Real local Ollama generation' if res['generated'] else 'Evidence-only, no LLM invocation')
        return f"**Mode:** {label}\n\n{res['answer']}",refs
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
    return (f"**{meta.get('mode','Unknown')}** · {meta.get('algorithm','')} · "
            f"{meta.get('documents','?')} documents · runtime {meta.get('runtime_seconds','?')} seconds",
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
    root=home()/'benchmarks'
    files=sorted(root.rglob('*_benchmark_*p.json'))
    choices=[str(p.relative_to(root)) for p in files]
    return gr.update(choices=choices,value=choices[-1] if choices else None)


def show_benchmark(filename):
    if not filename:
        return 'No real MPI benchmark files found. Run `python -m scripts.benchmark` inside WSL2.',None
    root=(home()/'benchmarks').resolve()
    path=(root/filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return 'Benchmark file not found.',None
    record=json.loads(path.read_text(encoding='utf-8'))
    runs=record['runs']
    p=record['mpi_processes']
    seq=[r['sequential_wall_seconds'] for r in runs]
    mpi=[r['mpi_end_to_end_wall_seconds'] for r in runs]
    fig=go.Figure()
    fig.add_trace(go.Bar(name='Sequential',x=[f'Run {i+1}' for i in range(len(runs))],y=seq))
    fig.add_trace(go.Bar(name=f'Real MPI ({p} processes)',x=[f'Run {i+1}' for i in range(len(runs))],y=mpi))
    fig.update_layout(barmode='group',template='plotly_white',height=340,yaxis_title='End-to-end seconds',
                      title='Actual measured runtime (includes process startup)')
    speed = (record.get('median_speedup') or
             statistics.median(seq)/statistics.median(mpi))
    unit=record.get('work_unit_label','work units')
    throughput=record.get('median_mpi_work_units_per_second')
    perf=(f' · **{throughput:,.0f} {unit}/s**' if throughput is not None else '')
    kind=record.get('dataset_kind','Input provenance not recorded — inspect the source dataset')
    return (f"**Real MPI measurement** · **Input: {kind}** · {record['operation']} · {record['documents']} documents · "
            f"{p} MPI processes · {len(runs)} repetitions · median-based speedup **{speed:.3f}×**"
            +perf+' · Actual rank evidence recorded. ' +
            'MPI slowdown is valid. Correctness verified on every saved run.'),fig


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
            return f"**Local Ollama-generated explanation (model: {output['model']})**\n\n{output['text']}\n\nSource references: see input document manifest."
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
            out=ingest_file(source, industry='SYNTHETIC / '+label)
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
