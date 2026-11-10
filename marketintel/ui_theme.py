"""Single design system for the four-page Market Intelligence research workspace.

Presentation only: this module does not import analytics, RAG, or MPI components.
"""

TITLE = 'AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS'

MASTHEAD = '''
<header id="mi-masthead" aria-label="Project introduction">
  <div class="mi-hero-top">
    <div class="mi-mark" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
    <div class="mi-brand">MARKET <strong>ATLAS</strong><span class="mi-brand-detail">/ RESEARCH SYSTEM</span></div>
    <span class="mi-environment"><span class="mi-beacon"></span> LOCAL RESEARCH BUILD</span>
  </div>
  <div class="mi-hero-main">
    <div>
      <div class="mi-eyebrow mi-eyebrow-light">RESEARCH INTELLIGENCE · COMPUTATIONAL ANALYTICS</div>
      <h1>AI-Powered Market Intelligence &amp; Document Analysis using RAG + HPC on AWS</h1>
      <p class="mi-hero-tagline">Discover the evidence. <strong>Prove the computation.</strong></p>
      <p>Ask questions across market reports, analyze document relationships and inspect exactly how every result was produced.</p>
    </div>
    <div class="mi-hero-aside" aria-label="Processing pipeline">
      <div class="mi-hero-aside-label">ANALYSIS WORKFLOW</div>
      <div>01 <span>REPORTS &amp; SOURCES</span></div>
      <div>02 <span>RETRIEVAL / COMPUTATION</span></div>
      <div>03 <span>RESULTS &amp; PROVENANCE</span></div>
    </div>
  </div>
  <div class="mi-full-title">DOCUMENTS  /  RETRIEVAL  /  PARALLEL ANALYTICS  /  EXPLAINABLE RESULTS</div>
</header>
'''

WORKFLOW = '''
<div class="mi-workflow" aria-label="Research workflow">
  <div><b>01</b><span>Ingest</span><small>PDF · DOCX · CSV</small></div>
  <div class="mi-workflow-arrow" aria-hidden="true">→</div>
  <div><b>02</b><span>Understand</span><small>Evidence retrieval</small></div>
  <div class="mi-workflow-arrow" aria-hidden="true">→</div>
  <div><b>03</b><span>Compute</span><small>Sequential / real MPI</small></div>
  <div class="mi-workflow-arrow" aria-hidden="true">→</div>
  <div><b>04</b><span>Explain</span><small>Results with sources</small></div>
</div>
'''

OVERVIEW_CARDS = '''
<div class="mi-split">
  <article class="mi-explain mi-explain-rag">
    <span class="mi-explain-tag">EVIDENCE ENGINE</span>
    <h3>Document intelligence</h3>
    <p>Read uploaded reports, retrieve the passages behind your answer and inspect their original sources.</p>
    <div class="mi-explain-foot">Source-first · TF-IDF locally · Optional local LLM</div>
  </article>
  <article class="mi-explain mi-explain-hpc">
    <span class="mi-explain-tag">COMPUTE ENGINE</span>
    <h3>Parallel market analytics</h3>
    <p>Measure report similarity and discover clusters; use real MPI only when an actual MPI runtime is available.</p>
    <div class="mi-explain-foot">Traceable vectors · Distributed algorithms · Honest benchmarks</div>
  </article>
</div>
'''

HPC_STEPS = '''
<div class="mi-mini-flow" aria-label="HPC job pipeline">
  <span><b>01</b> Document vectors</span>
  <span>→</span>
  <span><b>02</b> MPI partitions</span>
  <span>→</span>
  <span><b>03</b> Numerical results</span>
  <span>→</span>
  <span><b>04</b> Source-verified report</span>
</div>
'''

FOOTER = '''
<footer class="mi-footer">
  <span>MARKET ATLAS / ENGINEERING EDITION</span>
  <span>Local prototype · AWS integration pending · No simulated HPC results</span>
</footer>
'''

CSS = r'''
:root {
  --mi-ink:#132432;--mi-ink-light:#3c5661;--mi-mute:#718189;
  --mi-teal:#146e68;--mi-highlight:#bfdfca;--mi-rust:#cf8b54;
  --mi-border:#dce4e2;--mi-canvas:#f5f7f5;--mi-paper:#ffffff;
}
html, body {background:var(--mi-canvas)!important}
.gradio-container{max-width:1270px!important;margin:0 auto!important;padding:16px 24px 48px!important;
  background:var(--mi-canvas)!important;
  font-family:Inter,'Segoe UI',system-ui,sans-serif!important;color:var(--mi-ink)!important}
.gradio-container h1,.gradio-container h2,.gradio-container h3 {letter-spacing:-.035em}
.gradio-container :is(button, input, select, textarea){font-family:inherit!important}
#mi-masthead{border-radius:18px;background:#172d35;color:#fff;overflow:hidden;position:relative;
  border:1px solid #243b42;margin-bottom:14px;box-shadow:0 16px 35px rgba(24,48,56,.11)}
#mi-masthead:before{content:'';pointer-events:none;position:absolute;inset:0;opacity:.14;
  background-image:linear-gradient(90deg,transparent 97%,#8ca8a8 98%),linear-gradient(transparent 97%,#8ca8a8 98%);
  background-size:44px 44px;mask-image:linear-gradient(100deg,transparent 15%,#000)}
.mi-hero-top{position:relative;display:flex;align-items:center;gap:11px;padding:17px 26px;border-bottom:1px solid #ffffff20}
.mi-mark{display:grid;grid-template-columns:repeat(2,8px);gap:4px;transform:rotate(-10deg);margin-right:2px}
.mi-mark span{width:8px;height:8px;border-radius:2px;background:#afd4c3}.mi-mark span:nth-child(2){opacity:.5}
.mi-mark span:nth-child(3){background:#e8a26a}.mi-mark span:nth-child(4){opacity:.8}
.mi-brand{font-size:12px;letter-spacing:.18em;font-weight:650;color:#f4f7f3!important}.mi-brand strong{color:#e4bd89!important}.mi-brand-detail{color:#b1c6c7!important;font-size:10px;margin-left:12px;letter-spacing:.12em}
.mi-environment{margin-left:auto;border:1px solid #7c9d9c79;border-radius:100px;padding:6px 10px;font-size:10px;letter-spacing:.11em;color:#d2e7df;white-space:nowrap}
.mi-beacon{display:inline-block;background:#a8d5bf;width:6px;height:6px;border-radius:100%;margin-right:6px}
.mi-hero-main{position:relative;padding:26px 30px 24px;display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:28px;align-items:center}
.mi-eyebrow{font-size:11px;letter-spacing:.15em;font-weight:750;color:#32786d}.mi-eyebrow-light{color:#b6d9c9!important}
.mi-hero-main h1{font-size:clamp(24px,2.45vw,33px)!important;line-height:1.2!important;letter-spacing:-.04em!important;
  margin:10px 0 12px!important;color:#f9f5ea!important;font-weight:750!important}
.mi-hero-tagline,.mi-hero-tagline strong{color:#e1bd8b!important;font-weight:650;font-size:14px!important;margin-bottom:4px!important}
.mi-hero-main p{color:#c2d1d1;font-size:13px;line-height:1.65;max-width:680px}
.mi-hero-aside{border-left:1px solid #ffffff32;padding:8px 0 8px 22px;color:#91b7b1;font-family:ui-monospace,monospace;font-size:12px}
.mi-hero-aside-label{font-family:inherit;letter-spacing:.12em;font-size:10px;margin-bottom:11px;color:#d8bf9f}
.mi-hero-aside div:not(:first-child){border-bottom:1px solid #ffffff1d;padding:9px 0;white-space:nowrap;color:#d8bf9f!important}
.mi-hero-aside div span{font-family:Inter,'Segoe UI',sans-serif;color:#e5ebe7!important;margin-left:10px;font-size:11px;letter-spacing:.04em}
.mi-full-title{position:relative;background:#213943;padding:10px 30px;color:#c9d7d2;font-size:11px;letter-spacing:.015em}
#mi-nav{border-radius:14px;padding-top:2px}
#mi-nav .tab-wrapper{border-bottom:1px solid var(--mi-border)!important;margin-bottom:14px!important;gap:4px!important}
#mi-nav [role=tab]{border-radius:9px 9px 0 0!important;border:0!important;font-weight:650!important;font-size:13px!important;padding:12px 17px!important;color:#677981!important}
#mi-nav [role=tab].selected{color:#145f5e!important;background:#e5f1eb!important;box-shadow:inset 0 -3px 0 #1f7d74!important}
.gradio-container .header-table th.header-cell{background:#e8f1ee!important;color:#17343d!important;border-color:#d2e2dc!important}
.gradio-container .header-table th.header-cell *{color:#17343d!important}
.gradio-container table.file-preview tr.file{background:#e8f1ee!important;color:#17343d!important}
.gradio-container table.file-preview td.filename *{color:#17343d!important}
.mi-page-header{padding:5px 0 16px}
.mi-page-header h2{font-size:26px!important;line-height:1.25!important;color:var(--mi-ink)!important;margin:5px 0 7px!important}
.mi-page-header p{color:#60737a!important;font-size:13px;line-height:1.6;max-width:770px}
.mi-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:2px 0 14px}
.mi-kpi{background:#fff;border:1px solid var(--mi-border);padding:17px 20px;border-radius:12px;display:flex;flex-direction:column;min-height:122px}
.mi-kpi-label{color:#5f747a;font-size:12px;font-weight:620}.mi-kpi strong{font-size:29px;line-height:1.45;letter-spacing:-.06em;color:#193947}
.mi-kpi small{font-size:11px;color:#809297;margin-top:auto}
.mi-data-note{border-left:3px solid #237f74;background:#eaf4ef;border-radius:6px;padding:12px 16px;color:#3f6864;font-size:12px;margin-bottom:13px}
.mi-workflow{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:15px 17px;border:1px solid var(--mi-border);border-radius:12px;background:#fff;margin:14px 0 20px}
.mi-workflow>div:not(.mi-workflow-arrow){display:flex;flex-direction:column;gap:3px;min-width:0}
.mi-workflow b{font-size:10px;letter-spacing:.07em;color:#a07050}.mi-workflow span{font-size:14px;font-weight:650;color:#193540}.mi-workflow small{font-size:11px;color:#85969b}
.mi-workflow-arrow{color:#8fa6a1;font-size:24px}
.mi-split{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:12px 0 14px}
.mi-explain{padding:22px;border-radius:12px;border:1px solid #dce5e1;background:#fff;position:relative;min-height:200px}
.mi-explain-rag{border-top:4px solid #3d8276}.mi-explain-hpc{border-top:4px solid #c88d5d}
.mi-explain-tag{font-size:10px;font-weight:700;letter-spacing:.11em;color:#568079}.mi-explain h3{margin:10px 0 6px!important;color:#17323d}
.mi-explain p{font-size:13px;line-height:1.7;color:#5e737a}
.mi-explain-foot{margin-top:18px;font-size:11px;color:#8a7061}
.mi-mini-flow{border:1px solid var(--mi-border);border-radius:10px;background:#f1f6f3;padding:12px 15px;display:flex;align-items:center;justify-content:space-between;gap:8px;margin:12px 0 16px;flex-wrap:wrap;font-size:12px;color:#38565b}
.mi-mini-flow span{color:#38565b!important}
.mi-mini-flow b{color:#ba7f51;margin-right:6px}
.mi-tip{background:#e9f3ee;border-left:3px solid #287f72;border-radius:6px;padding:10px 14px;font-size:12px;color:#315850}
.mi-warning{background:#fbf3e9;border-left:3px solid #c89257;border-radius:6px;padding:10px 14px;font-size:12px;color:#765b3b}
.mi-footer{border-top:1px solid var(--mi-border);margin-top:24px;padding:16px 2px 8px;display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;font-size:10px;letter-spacing:.06em;color:#89969a}
.gradio-container .primary{font-weight:650!important}
@media(max-width:800px){.mi-hero-main{grid-template-columns:1fr}.mi-hero-aside{display:none}.mi-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.mi-split{grid-template-columns:1fr}.mi-workflow{flex-wrap:wrap}.mi-workflow-arrow{display:none}}
@media(max-width:500px){.gradio-container{padding:9px!important}.mi-hero-top{padding:14px 15px}.mi-hero-main{padding:22px 17px}.mi-full-title{padding:11px 17px}.mi-kpis{gap:7px}.mi-kpi{padding:12px}.mi-environment{font-size:8px}.mi-brand-detail{display:none}}
'''
