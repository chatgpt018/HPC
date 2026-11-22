"""Build offline charts strictly from a saved, correctness-checked MPI study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.graph_objects as go


def create_charts(study_path: Path, output: Path) -> Path:
    study = json.loads(study_path.read_text(encoding='utf-8'))
    rows = study.get('rows', [])
    if not rows or not all(row.get('verified') is True for row in rows):
        raise ValueError('Charts require actual saved configurations with verified MPI correctness')
    sizes = sorted({int(row['documents']) for row in rows})
    ranks = sorted({int(row['mpi_processes']) for row in rows})
    operations = sorted({row['operation'] for row in rows})
    largest = sizes[-1]
    figures: list[tuple[str, go.Figure]] = []
    colors = {'similarity': '#007f83', 'clustering': '#bd6534'}

    for title, field, ytitle, percent in (
        ('Execution time versus process count', 'median_mpi_seconds', 'End-to-end seconds', False),
        ('Speedup versus process count', 'speedup', 'Sequential / MPI time', False),
        ('Efficiency versus process count', 'efficiency_percent', 'Percent', True),
        ('Throughput versus process count', 'mpi_work_units_per_second', 'Work units / second', False),
    ):
        fig = go.Figure()
        for operation in operations:
            selected = sorted((r for r in rows if r['operation'] == operation and
                               int(r['documents']) == largest), key=lambda r: int(r['mpi_processes']))
            fig.add_trace(go.Scatter(x=[r['mpi_processes'] for r in selected],
                                     y=[r[field] for r in selected], name=operation,
                                     mode='lines+markers', line_color=colors[operation]))
        fig.update_layout(title=f'{title} · {largest} documents', xaxis_title='Actual MPI ranks',
                          yaxis_title=ytitle)
        fig.update_xaxes(tickvals=ranks)
        if percent:
            fig.update_yaxes(ticksuffix='%')
        figures.append((title, fig))

    runtime = go.Figure()
    memory = go.Figure()
    for operation in operations:
        color = colors[operation]
        sequential = [next(r for r in rows if r['operation'] == operation and
                           int(r['documents']) == size and int(r['mpi_processes']) == ranks[0])
                      for size in sizes]
        runtime.add_trace(go.Scatter(x=sizes,
                                     y=[r['median_sequential_seconds'] for r in sequential],
                                     name=f'{operation} · Sequential', mode='lines+markers',
                                     line=dict(color=color, dash='dot')))
        for rank in ranks:
            selected = [next(r for r in rows if r['operation'] == operation and
                             int(r['documents']) == size and int(r['mpi_processes']) == rank)
                        for size in sizes]
            runtime.add_trace(go.Scatter(x=sizes, y=[r['median_mpi_seconds'] for r in selected],
                                         name=f'{operation} · {rank} ranks', mode='lines+markers',
                                         line=dict(color=color, dash='solid' if rank == ranks[0] else 'dash')))
            memory.add_trace(go.Scatter(
                x=sizes, y=[r['median_mpi_rank_peak_rss_kib'] / 1024 for r in selected],
                name=f'{operation} · {rank} ranks', mode='lines+markers',
                line=dict(color=color, dash='solid' if rank == ranks[0] else 'dash')))
    runtime.update_layout(title='Runtime versus dataset size', xaxis_title='Synthetic documents',
                          yaxis_title='End-to-end seconds')
    memory.update_layout(title='Peak memory versus dataset size', xaxis_title='Synthetic documents',
                         yaxis_title='Median peak rank RSS · MiB')
    figures.extend([('Runtime versus dataset size', runtime), ('Memory versus dataset size', memory)])

    breakdown = go.Figure()
    selected = [r for r in rows if int(r['documents']) == largest]
    labels = [f"{r['operation']} · {r['mpi_processes']} ranks" for r in selected]
    breakdown.add_trace(go.Bar(x=labels, y=[r['median_mpi_section_seconds'] for r in selected],
                               name='Measured MPI section', marker_color='#007f83'))
    breakdown.add_trace(go.Bar(x=labels, y=[r['median_mpi_outside_section_seconds'] for r in selected],
                               name='Outside section: startup, load, output', marker_color='#bd6534'))
    breakdown.update_layout(title=f'MPI section versus outside-section time · {largest} documents',
                            yaxis_title='Median seconds', barmode='stack')
    figures.append(('Timing breakdown', breakdown))

    panels = []
    for index, (title, figure) in enumerate(figures):
        figure.update_layout(template='plotly_white', height=410,
                             font=dict(family='Arial', size=13, color='#102f3c'),
                             paper_bgcolor='white', plot_bgcolor='white',
                             margin=dict(l=70, r=30, t=65, b=70))
        panels.append(f'<section><h2>{title}</h2>' + figure.to_html(
            full_html=False, include_plotlyjs='inline' if index == 0 else False,
            config={'displaylogo': False}) + '</section>')
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Measured MPI scaling study</title><style>body{font:16px Arial;'
            'max-width:1100px;margin:32px auto;padding:0 20px;color:#102f3c}'
            'section{margin:30px 0;border:1px solid #dce7e5;border-radius:10px;'
            'padding:14px 20px}h1{font-size:28px}h2{font-size:18px}'
            'p{line-height:1.5}</style><h1>Genuine MPI scaling study</h1>'
            '<p>Measured synthetic inputs on one host. All plotted MPI outputs passed '
            'independent Sequential correctness checks. Three repetitions per '
            'configuration; medians include process launch and file I/O. '
            'Memory is a per-rank peak, not total cluster memory. The timing '
            'breakdown does not isolate communication from process startup.</p>'
            + ''.join(panels) + '</html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(create_charts(args.study, args.output))


if __name__ == '__main__':
    main()
