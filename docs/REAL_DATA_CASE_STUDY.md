# Build a legitimate market intelligence case study

Do not label fictional example reports as real industry findings.

## Suggested public starting points

- [U.S. SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces): annual/quarterly company filings; respect fair-access rules and keep filing year/company/source URL.
- [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392): economic time series such as GDP or energy-related indicators; economic indicators are not automatically market-size estimates.
- [U.S. EIA Open Data](https://www.eia.gov/opendata/): energy indicators and time series; an API key may be required and the published usage terms apply.

Use the links to **obtain and review data yourself**. No live external data is bundled or downloaded by the application. Make sure you have permission to store and demonstrate each report.

## Import a sourced case study

1. Obtain a small, permitted set of reports yourself and record the licence or
   usage terms. Put PDF, DOCX, or CSV files in `data/raw/reports/`; this folder
   is ignored by Git and absent from the release ZIP.
2. Copy `data/metadata/manifest.example.csv` to `data/metadata/manifest.csv`.
   Add one row per file. Use a filename only, without a directory. Complete
   `publisher`, `publication_date` (`YYYY-MM-DD`), `usage_notes`, and
   `source_kind=real`. Record the source URL when the file came from the web;
   the reporting period and geographic scope prevent misleading comparisons.
3. Run `python -m marketintel.cli ingest-manifest data/metadata/manifest.csv`.
   The command validates every manifest row before indexing. The project stores
   a full content SHA-256 and source metadata with each document, then includes
   those fields in analysis provenance. Reimporting identical bytes adds no
   duplicate passages. A conflicting attribution is rejected.
4. Open Documents & AI to inspect extracted source locations. Check at least
   five passages against originals. Ask a specific source question, then use
   Sequential similarity or clustering on at least two distinct reports.
5. Record the document IDs and dataset fingerprint from Results & Reports.
   Run actual MPI separately where available and compare with Sequential.

For retrieval evaluation, copy `data/metadata/rag_evaluation.example.csv` to a
private CSV. Add manually checked questions and the expected indexed document
ID plus page/paragraph/row for each answerable question. Repeat a question
for multiple required sources. Leave both expected fields blank for an
insufficient-evidence question. Include factual, multi-document, comparison,
misleading-assumption, and insufficient-evidence cases. Then run:

```text
python -m scripts.evaluate_retrieval --questions data/metadata/rag_evaluation.csv --top-k 4
```

The command reports actual Hit@k, Recall@k, and no-evidence accuracy for the
current index. It refuses labels that do not point to indexed source locations.
It does not score LLM truthfulness, and no real-data scores are prefilled.

## Review a generated answer before citing it

When a local embedding model and Ollama model are installed, save the question,
retrieval mode, model name, answer, and retrieved passage IDs. For each factual
claim, open its numbered passage in the original report and mark **supported**,
**unsupported**, or **unclear**. Recheck every number, unit, year, geography,
and calculation against the source. Record missing relevant passages and whether
the model correctly abstained when evidence was insufficient. A valid citation
number only proves the passage was retrieved; it does not prove the claim is
true. Do not report a generated answer as verified until a person completes
this review. No live model review has been completed for this release.

No real documents are bundled. The practice collection and synthetic scaling
vectors remain teaching and performance inputs, never real market evidence.

## Suggested minimum evaluated dataset

- 20–50 genuine reports initially, grouped by company or industry and including publication year.
- The bundled manifest template records publisher, date, source URL, reporting
  period, geography, usage terms, and an explicit real/synthetic classification.
- Optional CSV time series with exact columns: `market,year,value,unit,region,currency`. Do not combine incompatible currencies, units, date ranges or geographies.
- Run document ingestion and verify 5 random passages against original PDFs before using them in the demonstration.
- The MPI scaling study uses **separately labeled synthetic vectors** for reproducible large N experiments. Do not clone/duplicate reports to exaggerate unique document counts.

## Present a defensible case study

Select a narrow question, such as *Which uploaded semiconductor reports discuss supply-chain capacity?* Use cited retrieval for factual evidence. Use similarity and clustering to discover related reports; **do not claim the resulting clusters prove future growth**. Use real numeric time series only for valid CAGR and YoY calculations.
