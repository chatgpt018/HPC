# Dataset plan and data integrity

**Project:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Two separate dataset purposes

1. **Real source documents:** Start with 20–50 public reports (company filings,
   government/industry reports), and increase gradually. Preserve original
   filenames and document-level source details. SEC EDGAR filings and publicly
   downloadable government reports are suitable sources, subject to reuse terms.
2. **Controlled HPC benchmarks:** Use the *explicitly synthetic* generator in
   `scripts/generate_scaling_data.py`. Repeat the same numerical workload for
   all process counts; do not pretend these vectors represent real markets.

For numerical market metrics, upload CSV with precisely:

```
market,year,value,unit,region,currency
```

The application calculates CAGR/YoY only when a market series is comparable,
with positive values and consistent region, currency and unit. Do not infer
market-size trends from text embeddings or clusters.

## Suggested evaluation datasets

- Small smoke case: 3–20 reports. Verify source locations, no missing IDs.
- Functional case: 20–100 genuine reports from different sectors.
- HPC scaling: 500/1000/2500/5000+ synthetic vectors, dimension recorded.
  Choose sizes based on available memory and realistic runtime.
- Final report-based analysis: the same actual uploaded documents must drive
  the similarity and clustering results shown in the dashboard.

## Research and attribution

- Include publisher, release date, document URL and a statement of permitted
  download or reuse for each real report in your submission's dataset appendix.
- Check that any forecast is labelled as a forecast, not a historical fact.
- Keep confidential documents and original paid reports out of Git/public ZIPs.
- Market clusters do **not** establish causal relationships or investment return.
