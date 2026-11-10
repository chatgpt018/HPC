"""Market Atlas: four-page Gradio research interface and data quality studio.

Contains composition and layout only. Data operations live in ui_actions.py;
visual identity lives in ui_theme.py. The submitted project title is unchanged.
"""
from __future__ import annotations

import os

import gradio as gr

from .ui_actions import (
    overview, upload, table_documents, ask, run_analysis, list_results,
    inspect_result, result_chart, benchmark_options, show_benchmark,
    display_market_metrics, explain_saved_result, download_brief, download_result,
    _latest_results, workspace_quality, import_practice_collection, compute_preview,
    document_options, document_overview, analysis_guidance,
)
from .ui_theme import TITLE, CSS, MASTHEAD, WORKFLOW, OVERVIEW_CARDS, HPC_STEPS, FOOTER
from .storage import records


def page_intro(number: str, eyebrow: str, title: str, description: str) -> None:
    """Render consistent editorial section introductions using fixed local text."""
    import html
    gr.HTML(f'''<div class="mi-page-header"><div class="mi-eyebrow">{html.escape(number)} / {html.escape(eyebrow)}</div>
      <h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>''')


def build() -> gr.Blocks:
    """Build a functional, local-first UI with no simulated MPI success states."""
    with gr.Blocks(
        title=TITLE,
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        gr.HTML(MASTHEAD)
        with gr.Tabs(elem_id='mi-nav'):
            with gr.Tab('01  Overview', id='overview'):
                page_intro('01', 'WORKSPACE', 'Your research, in one place',
                           'A focused overview of the documents, computation and evidence available in this local project.')
                summary = gr.HTML()
                gr.HTML(WORKFLOW)
                gr.HTML(OVERVIEW_CARDS)
                with gr.Accordion('Research coverage · inspect real source quality', open=False):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            quality_note = gr.Markdown()
                        with gr.Column(scale=7):
                            quality_chart = gr.Plot(label='Industry coverage')
                    refresh_quality = gr.Button('↻ Update evidence coverage', variant='secondary', size='sm')
                    refresh_quality.click(workspace_quality, outputs=[quality_note, quality_chart])
                    app.load(workspace_quality, outputs=[quality_note, quality_chart])
                with gr.Accordion('Environment and implementation status', open=False):
                    environment = gr.Markdown()
                    gr.Markdown('**Not yet verified:** live multi-rank MPI execution, real local Ollama inference, and AWS deployment. '
                                'A saved sample or a working sequential algorithm does not establish these capabilities.')
                with gr.Row():
                    refresh = gr.Button('↻ Refresh workspace counts', variant='secondary')
                refresh.click(overview, outputs=[summary, environment])
                app.load(overview, outputs=[summary, environment])

            with gr.Tab('02  Documents & AI', id='documents'):
                page_intro('02', 'DOCUMENT INTELLIGENCE', 'Bring your research. Find the evidence.',
                           'Index a document, read its extracted text, then ask about a topic you can see in it.')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=280):
                        gr.Markdown('#### 01 / Add your source documents')
                        uploads = gr.File(
                            file_types=['.pdf', '.docx', '.csv'], file_count='multiple', type='filepath',
                            label='Choose PDF, Word or CSV reports', elem_id='mi-file-upload', height=160,
                        )
                        industry = gr.Textbox(
                            value='Unspecified', label='Industry (optional)',
                            placeholder='e.g. Renewable energy',
                        )
                        upload_btn = gr.Button('Index documents', variant='primary')
                        with gr.Accordion('No reports yet? Load practice data', open=False):
                            gr.Markdown('Nine short **SYNTHETIC** teaching reports are bundled with this app. '
                                        'They are never represented as factual industry research.')
                            demo_btn = gr.Button('Load clearly labeled synthetic examples', variant='secondary')
                        log = gr.Textbox(label='Import result', lines=3, interactive=False)
                    with gr.Column(scale=7, min_width=360):
                        gr.Markdown('#### 02 / Indexed library')
                        documents = gr.Dataframe(
                            headers=['Source document', 'Industry', 'Passages'],
                            datatype=['str', 'str', 'number'],
                            interactive=False, wrap=False, column_widths=['65%', '20%', '15%'],
                        )
                        document_refresh = gr.Button('↻ Refresh document list', size='sm', variant='secondary')
                        gr.HTML('<div class="mi-tip">Passages are pieces of extracted text, not page counts. '
                                'Every passage keeps its source location and document ID.</div>')
                gr.Markdown('#### 03 / Read what was extracted')
                indexed = [(d['filename'], d['doc_id']) for d in records()]
                selected_id = indexed[-1][1] if indexed else None
                document_choice = gr.Dropdown(choices=indexed, value=selected_id,
                                              label='Indexed document to inspect')
                preview = gr.HTML(document_overview(selected_id))

                def refresh_document_view():
                    return table_documents(), document_options(), document_overview()

                def index_and_preview(files, sector):
                    feedback, rows = upload(files, sector)
                    return feedback, rows, document_options(), document_overview()

                def practice_and_preview():
                    feedback, rows = import_practice_collection()
                    return feedback, rows, document_options(), document_overview()

                document_refresh.click(refresh_document_view, outputs=[documents, document_choice, preview])
                app.load(refresh_document_view, outputs=[documents, document_choice, preview])
                upload_btn.click(index_and_preview, inputs=[uploads, industry],
                                 outputs=[log, documents, document_choice, preview])
                demo_btn.click(practice_and_preview, outputs=[log, documents, document_choice, preview])
                document_choice.change(document_overview, inputs=document_choice, outputs=preview)
                gr.Markdown('---')
                gr.Markdown('#### 04 / Ask about a topic in the document')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        question = gr.Textbox(
                            label='Research question', lines=3,
                            placeholder='What risks or opportunities are described in the uploaded energy reports?',
                        )
                        ask_btn = gr.Button('Find evidence', variant='primary')
                        with gr.Accordion('Optional local AI generation', open=False):
                            gen = gr.Checkbox(
                                label='Generate an answer using an already installed local Ollama model',
                                value=False,
                            )
                            gr.Markdown('Without Ollama, the result is **evidence-only**, not a generated LLM answer. '
                                        'Enabling this option does not make a cloud API call.')
                    with gr.Column(scale=5):
                        gr.HTML('<div class="mi-tip"><strong>How to read an answer</strong><br>'
                                '1. Read the response.<br>2. Check the supporting excerpts.<br>'
                                '3. Open the cited source before making a market claim.</div>')
                answer_output = gr.Markdown('Choose a topic from the document overview above, then click Find evidence.')
                gr.Markdown('#### Retrieved evidence')
                evidence_output = gr.Dataframe(
                    headers=['Document', 'Source location', 'Relevance', 'Excerpt'],
                    datatype=['str', 'str', 'number', 'str'], interactive=False, wrap=True,
                    column_widths=['250px', '120px', '90px', '600px'], visible=False,
                )
                def find_and_show(question_text, use_model):
                    summary, rows = ask(question_text, use_model)
                    return summary, gr.update(value=rows, visible=bool(rows))

                ask_btn.click(find_and_show, inputs=[question, gen], outputs=[answer_output, evidence_output])
                question.submit(find_and_show, inputs=[question, gen], outputs=[answer_output, evidence_output])

            with gr.Tab('03  HPC Analytics', id='hpc') as hpc_tab:
                page_intro('03', 'COMPUTATIONAL ANALYSIS', 'From reports to measurable relationships',
                           'Calculate exact document similarity or clusters using features derived from uploaded reports. '
                           'Choose real MPI only when a working MPI runtime is installed.')
                gr.HTML(HPC_STEPS)
                with gr.Accordion('Why use HPC here? · measured workload, not marketing', open=False):
                    work_preview = gr.Markdown()
                    gr.Button('Compute current workload size', variant='secondary', size='sm').click(
                        compute_preview, outputs=work_preview)
                    app.load(compute_preview, outputs=work_preview)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        gr.Markdown('#### Configure the analysis')
                        operation = gr.Dropdown(
                            ['Exact similarity · top-k', 'K-Means clustering'],
                            value='Exact similarity · top-k', label='Analysis method',
                        )
                        mode = gr.Radio(['Sequential', 'Real MPI'], value='Sequential', label='Execution method')
                        with gr.Row():
                            count = gr.Slider(1, 10, value=3, step=1, label='Top matches per document')
                            procs = gr.Slider(2, 8, value=2, step=1, label='MPI processes', visible=False)
                        with gr.Accordion('Vector representation', open=False):
                            vector_mode = gr.Radio(
                                ['hashing', 'semantic'], value='hashing', label='Numerical input',
                            )
                            gr.Markdown('**Hashing:** reproducible lexical features. **Semantic:** requires an '
                                        'already installed local neural model. These are not interchangeable.')
                        initial_guidance, _ = analysis_guidance('Exact similarity · top-k', 3, 'Sequential')
                        guidance = gr.Markdown(initial_guidance, elem_id='mi-analysis-guidance')
                        start = gr.Button('Run computed analysis', variant='primary',
                                          interactive=len(indexed) >= 2)
                    with gr.Column(scale=5):
                        gr.HTML('''<div class="mi-warning"><strong>Computation integrity</strong><br>
                        • Sequential mode never claims MPI execution.<br>
                        • Real MPI must launch multiple ranks and save rank metadata.<br>
                        • No hardcoded speedups, job IDs, or market forecasts.<br>
                        • Dataset must contain uploaded documents.</div>''')
                        gr.Markdown('#### What is measured?')
                        gr.Markdown('**Similarity:** cosine scores between document feature vectors.  '
                                    '**Clustering:** groups of document features, not predicted industry performance.  '
                                    '**Performance:** actual elapsed computation time; multi-rank benchmarking requires the MPI runtime.')
                gr.Markdown('#### Analysis output')
                report = gr.Markdown('A result table will appear here after a valid run.')
                show = gr.Dataframe(
                    headers=['Source document', 'Match / cluster', 'Score'],
                    interactive=False, wrap=True, visible=False,
                )
                with gr.Accordion('Technical run receipt · provenance and process metadata', open=False):
                    metadata = gr.Code(label='Actual run metadata', language='json')
                def run_and_show(*settings):
                    summary, rows, receipt = run_analysis(*settings)
                    return summary, gr.update(value=rows, visible=bool(rows)), receipt

                for control in (operation, count, mode):
                    control.change(analysis_guidance, inputs=[operation, count, mode],
                                   outputs=[guidance, start])
                hpc_tab.select(analysis_guidance, inputs=[operation, count, mode],
                               outputs=[guidance, start])
                app.load(analysis_guidance, inputs=[operation, count, mode],
                         outputs=[guidance, start])
                operation.change(
                    lambda choice: gr.update(label='Number of groups' if choice == 'K-Means clustering'
                                             else 'Top matches per document'),
                    inputs=operation, outputs=count,
                )
                mode.change(lambda choice: gr.update(visible=choice == 'Real MPI'),
                            inputs=mode, outputs=procs)
                start.click(
                    run_and_show,
                    inputs=[operation, mode, procs, count, vector_mode],
                    outputs=[report, show, metadata],
                )

            with gr.Tab('04  Results & Reports', id='results') as results_tab:
                page_intro('04', 'RESULTS STUDIO', 'Every finding, backed by a run',
                           'Inspect saved outputs, review the calculation receipt, download source-aware briefs and compare genuine MPI benchmarks.')
                gr.Markdown('#### Computed result explorer')
                with gr.Row():
                    saved = gr.Dropdown(
                        choices=[p.name for p in _latest_results()], label='Saved analysis', scale=5,
                    )
                    refresh_saved = gr.Button('↻ Refresh results', scale=1, min_width=150, variant='secondary')
                refresh_saved.click(list_results, outputs=saved)
                app.load(list_results, outputs=saved)
                results_tab.select(list_results, outputs=saved)
                saved_description = gr.Markdown('Select a saved analysis above, then inspect it. If the list is empty, index at least two documents and run an analysis first.')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        saved_table = gr.Dataframe(
                            headers=['Source document', 'Match / cluster', 'Score'],
                            interactive=False, wrap=True, visible=False,
                        )
                    with gr.Column(scale=6):
                        saved_plot = gr.Plot(label='Actual computed result', visible=False)
                with gr.Accordion('Behind the Analysis · actual calculation metadata', open=False):
                    saved_metadata = gr.Code(language='json', label='Source and algorithm receipt')
                def inspect_and_plot(name):
                    description, table, metadata = inspect_result(name)
                    chart = result_chart(name)
                    return (description, gr.update(value=table, visible=bool(table)), metadata,
                            gr.update(value=chart, visible=chart is not None))
                saved.change(inspect_and_plot, inputs=saved,
                             outputs=[saved_description, saved_table, saved_metadata, saved_plot])
                gr.Button('Inspect selected result', variant='primary').click(
                    inspect_and_plot, inputs=[saved],
                    outputs=[saved_description, saved_table, saved_metadata, saved_plot],
                )
                with gr.Accordion('Download the result and create a research brief', open=False):
                    with gr.Row():
                        with gr.Column():
                            download = gr.File(label='Computed result JSON', interactive=False)
                            gr.Button('Prepare JSON download', variant='secondary').click(
                                download_result, inputs=saved, outputs=download,
                            )
                        with gr.Column():
                            brief_download = gr.File(label='Source-aware research brief', interactive=False)
                            gr.Button('Prepare Markdown brief', variant='secondary').click(
                                download_brief, inputs=saved, outputs=brief_download,
                            )
                    explanation_ollama = gr.Checkbox(
                        value=False, label='Use already installed local Ollama model (optional)',
                    )
                    explanation = gr.Markdown()
                    gr.Button('Explain selected findings', variant='secondary').click(
                        explain_saved_result, inputs=[saved, explanation_ollama], outputs=explanation,
                    )
                    gr.Markdown('Without a configured local model this produces a deterministic source-aware brief, not AI generation.')
                with gr.Accordion('Market metrics · source-reported time series', open=False):
                    gr.Markdown('Upload a CSV with `market,year,value,unit,region,currency` columns. '
                                'Only compatible rows from actual files are used to calculate growth statistics.')
                    metric_status = gr.Markdown()
                    metric_table = gr.Dataframe(
                        headers=['Market', 'Region', 'Year', 'Value', 'Unit', 'Currency', 'YoY %', 'Series CAGR %', 'Source'],
                        interactive=False, wrap=True,
                    )
                    metric_plot = gr.Plot()
                    gr.Button('Calculate from uploaded CSV', variant='secondary').click(
                        display_market_metrics, outputs=[metric_status, metric_table, metric_plot],
                    )
                with gr.Accordion('Sequential vs real MPI · experimental benchmarks', open=False):
                    gr.Markdown('Benchmarks are displayed only when genuine experiments have been saved. '
                                'Negative speedup is a valid experimental result.')
                    with gr.Row():
                        benchmark_file = gr.Dropdown(choices=[], label='Saved real MPI benchmark', scale=5)
                        gr.Button('↻ Refresh benchmarks', scale=1, min_width=150, variant='secondary').click(
                            benchmark_options, outputs=benchmark_file,
                        )
                    benchmark_summary = gr.Markdown()
                    benchmark_plot = gr.Plot()
                    gr.Button('View measured performance', variant='secondary').click(
                        show_benchmark, inputs=benchmark_file,
                        outputs=[benchmark_summary, benchmark_plot],
                    )
                with gr.Accordion('Scientific limitations and interpretation', open=False):
                    gr.Markdown('''- Similarity is mathematical closeness, **not a market forecast**.
                    - Cluster numbers have no intrinsic industry meaning.
                    - Default TF-IDF is lexical retrieval; it is not a transformer RAG embedding model.
                    - Evidence-only answers are not LLM generation.
                    - Local MPI execution must be checked on a real MPI runtime with recorded ranks.
                    - AWS deployment is a separate later stage and is not claimed in this build.''')
        gr.HTML(FOOTER)
    return app


def main() -> None:
    # Bind locally by default. Do not expose this prototype directly to the internet.
    build().queue().launch(
        server_name=os.environ.get('MARKETINTEL_BIND', '127.0.0.1'),
        server_port=int(os.environ.get('MARKETINTEL_PORT', '7860')),
        inbrowser=False,
        css=CSS,
        theme=gr.themes.Soft(primary_hue='teal', neutral_hue='slate').set(
            body_background_fill='#f5f7f5', body_background_fill_dark='#f5f7f5',
            body_text_color='#132432', body_text_color_dark='#132432',
            background_fill_primary='#ffffff', background_fill_primary_dark='#ffffff',
            background_fill_secondary='#f1f6f3', background_fill_secondary_dark='#f1f6f3',
            block_background_fill='#ffffff', block_background_fill_dark='#ffffff',
            input_background_fill='#ffffff', input_background_fill_dark='#ffffff',
        ),
    )
