# Local container handoff (not an AWS deployment)

**AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS**

The `Dockerfile` packages only the web application. It does **not** provision Slurm,
AWS ParallelCluster or MPI compute nodes. It has no AWS SDK or cloud credentials.

```bash
docker build -t market-intelligence-local .
docker run --rm -p 127.0.0.1:7860:7860 \
  -v marketintel-data:/workspace/storage market-intelligence-local
```

Then open `http://127.0.0.1:7860` locally. Do **not** publish this unprotected
single-user application on the public internet. For public deployment add
identity/authentication, storage isolation, a durable job queue and a private
HPC submission service. Local MPI execution is **not** enabled in this web
container. The actual HPC environment needs a separately built MPI-compatible
image or compute-node environment, plus actual Slurm scheduling.

Docker may incur usage on your own machine but these commands do not make AWS calls.
