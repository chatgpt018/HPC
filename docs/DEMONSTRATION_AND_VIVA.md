# Your final-year live demonstration and viva plan

**Title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Presentation opening (30 seconds)

"We built a market intelligence and document-analysis platform. It reads PDF,
DOCX and CSV reports, retrieves supporting evidence, and calculates document
similarities and clusters. The HPC part is an actual MPI algorithm, not only
an HPC logo. We compare it with sequential computation and record numerical
correctness, runtime, speedup and parallel efficiency. The local version is the
foundation for our AWS deployment." (Use the last sentence only until real
AWS deployment is complete.)

## Suggested live demo (8 minutes)

1. **Overview (45 seconds):** Name the two paths: retrieval and numerical
   analytics. Point out truthful local/MPI/model status.
2. **Document upload (1 minute):** Upload three or more permitted real reports.
   Explain original file preservation and source location metadata.
3. **Evidence retrieval (1 minute):** Ask a question answerable by one report;
   show the exact passage and location. If real Ollama is configured, display
   the generated response while keeping the sources visible.
4. **Sequential HPC baseline (1 minute):** Run top-k similarity on those same
   documents. Point out the dataset SHA-256, document IDs and actual scores.
5. **Distributed algorithm (1–2 minutes):** Show a previously verified or
   currently running **real** `mpiexec` session with ranks, hostnames and
   partition ranges. Run or show K-Means `Allreduce` execution logs.
6. **Results (1 minute):** Explain that similarity is about numerical text
   features, not future market growth. Download a real source-aware brief.
7. **Benchmark (1 minute):** Compare actual sequential and MPI executions,
   including any negative speedup. Explain communication/startup overhead.
8. **AWS (30 seconds):** If deployed, show actual service and Slurm evidence;
   otherwise explain that AWS deployment is a remaining requirement.

## 20 viva questions and concise answers

1. **Why this title?** The application uses document retrieval, distributed
   computation and an AWS-oriented deployment architecture.
2. **Why HPC?** Pairwise similarity grows with the square of the document count;
   MPI divides query rows across processes.
3. **Why not simply use more threads?** MPI coordinates separate processes and
   supports multi-node distribution, unlike OpenMP alone.
4. **What is MPI rank?** A unique ID assigned to a process in the communicator.
5. **What is `COMM_WORLD`?** The communicator containing all launched MPI ranks.
6. **What is `gather`?** A collective collecting per-rank outputs at a root.
7. **What is `Allreduce`?** A collective combining values and returning the
   result to every participating rank.
8. **Does MPI always improve speed?** No. Small workloads suffer overhead.
9. **What is the reference implementation?** The sequential program defining
   correct similarity and K-Means output for comparison.
10. **How is cosine computed?** Dot product divided by both vector norms.
11. **Why normalize?** It makes cosine equal to a dot product for nonzero vectors.
12. **What is top-k?** The k most similar other documents per document.
13. **What does a cluster mean?** A group of vectors minimizing within-cluster
    squared distance under the chosen K-Means algorithm.
14. **Are cluster IDs industries?** No; they are arbitrary numbers without
    external validation or source-based labelling.
15. **Why keep document IDs?** For provenance from result rows to source files.
16. **What is TF-IDF?** A lexical term-weighting representation for retrieval.
17. **Is TF-IDF semantic neural retrieval?** No. A separately configured
    sentence-transformer is required for neural sentence embeddings.
18. **What is RAG?** Retrieve source evidence, then have an LLM generate an
    answer conditioned on that evidence.
19. **What does S3 do in the planned cloud architecture?** Store documents,
    input matrices and analytical results, with controlled permissions.
20. **How will you prove deployment?** With real web endpoint, Slurm job ID,
    MPI rank/node allocation, S3 output and a reproducible end-to-end test.

## What not to claim

Do not call synthetic benchmark vectors "real market reports". Do not claim
positive MPI speedup until measured. Do not call evidence-only retrieval an
LLM-generated answer. Do not claim AWS was deployed because configuration
files or a Dockerfile exist.
