# Algorithms: explain them during your viva

**Project:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## 1. Document to numbers

The uploaded PDF, DOCX and CSV reports are parsed into source-located chunks.
For the default local demonstration, each document is represented by a
**384-dimensional hashed lexical feature vector**. This is *not* a neural embedding.
An optional locally installed SentenceTransformer can generate real neural vectors.
The same document IDs are preserved from ingestion to saved analytical outputs.

## 2. Exact top-k cosine similarity

For two feature vectors A and B, cosine similarity is

`cos(A,B) = (A·B) / (||A|| × ||B||)`.

We normalize nonzero vectors to unit length and compare document rows against
all reference rows, excluding each document's self-match. Score ties are
resolved deterministically using document IDs. For `N` documents of dimension
`d`, exhaustive comparison performs approximately `O(N²d)` arithmetic. A
partition step finds each row's exact top-k cutoff; only selected scores and
cutoff ties are sorted. The block score buffer uses `O(block_size × N)` memory
instead of allocating an entire `N × N` similarity matrix. The reference
matrix is still `O(Nd)` per MPI rank in similarity mode.

**Worked example:** `A = (1, 0)` and `B = (0.6, 0.8)` both have unit norm, so
their cosine score is `1 × 0.6 + 0 × 0.8 = 0.6`. A vector `(0, 1)` has score
zero against `A`. The program excludes the document's own row from its top-k.

**MPI version:** rank 0 loads and normalizes the input matrix once, broadcasts
it to other ranks, and partitions query rows across actual ranks. Each rank
computes its own top-k results, then `MPI.gather` merges the output. This
parallelizes compute but *replicates the reference matrix* across ranks. It
should not be claimed to have perfect distributed-memory scaling.

## 3. Lloyd K-Means clustering

The goal is to minimize the within-cluster squared Euclidean distance (inertia)
for normalized document vectors. Select deterministic initial centroids. Repeat:
assign each point to its closest centroid, sum the vectors per cluster, divide
by cluster size and stop when centroid movement is below the tolerance.

**MPI version:** rank 0 loads and normalizes the matrix and scatters disjoint
row blocks. All ranks start from the same centroids. Each rank computes
assignments, sums and counts for its rows. `MPI.Allreduce` combines these
statistics so every rank computes the same global centroids. Final assignments
are gathered at rank 0. The implementation retains old centroids for empty
clusters. Cluster numbers are arbitrary labels.

**Worked partition:** seven vectors on three ranks give row spans `[0, 3)`,
`[3, 5)`, and `[5, 7)`. Each rank sums its assigned points by cluster;
`Allreduce` produces the same global sums and counts on every rank. Dividing
each sum by its count updates the centroid. If a count is zero, the old
centroid is retained. With more ranks than vectors, some spans are empty and
still participate in the collectives.

## 4. Correctness first

Compare sequential and MPI top-k pairs/scores with floating-point tolerance.
For clustering, check co-assignment groups (cluster IDs may be permuted),
centroids under the label permutation, inertia and iteration count. Carefully check uneven row partitions,
zero vectors, empty clusters and more ranks than rows.

## 5. Speedup is an experiment, not a promise

`S(p) = T(sequential) / T(MPI with p processes)`.

`Efficiency = 100 × S(p) / p`.

Use actual MPI ranks, valid outputs and repeated runs. The benchmarking tool
reports median wall-clock times **including process startup** plus separate
compute-only measurements. On small inputs MPI may be slower; report it.

## 6. Retrieval vs generation

TF-IDF selects source passages in the standard offline mode. This is lexical
retrieval, not transformer-based semantic retrieval. When an approved local
Ollama model is configured, it generates an answer using those passages.
The displayed source table lets a user independently check the original text.
With no Ollama model, only evidence retrieval is performed — do not call it a
completed LLM answer.
The optional neural index persists chunk vectors and re-encodes only changed
passages. Numbered model citations must refer to retrieved sources, though
this mechanical check cannot prove that each generated claim is true.
