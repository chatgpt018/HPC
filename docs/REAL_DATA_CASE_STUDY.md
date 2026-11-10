# Build a legitimate market intelligence case study

Do not label fictional example reports as real industry findings.

## Suggested public starting points

- [U.S. SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces): annual/quarterly company filings; respect fair-access rules and keep filing year/company/source URL.
- [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392): economic time series such as GDP or energy-related indicators; economic indicators are not automatically market-size estimates.
- [U.S. EIA Open Data](https://www.eia.gov/opendata/): energy indicators and time series; an API key may be required and the published usage terms apply.

Use the links to **obtain and review data yourself**. No live external data is bundled or downloaded by the application. Make sure you have permission to store and demonstrate each report.

## Suggested minimum evaluated dataset

- 20–50 genuine reports initially, grouped by company or industry and including publication year.
- A simple manifest with `file_name, publisher, title, year, source_url, category, license_or_terms`.
- Optional CSV time series with exact columns: `market,year,value,unit,region,currency`. Do not combine incompatible currencies, units, date ranges or geographies.
- Run document ingestion and verify 5 random passages against original PDFs before using them in the demonstration.
- The MPI scaling study uses **separately labeled synthetic vectors** for reproducible large N experiments. Do not clone/duplicate reports to exaggerate unique document counts.

## Present a defensible case study

Select a narrow question, such as *Which uploaded semiconductor reports discuss supply-chain capacity?* Use cited retrieval for factual evidence. Use similarity and clustering to discover related reports; **do not claim the resulting clusters prove future growth**. Use real numeric time series only for valid CAGR and YoY calculations.
