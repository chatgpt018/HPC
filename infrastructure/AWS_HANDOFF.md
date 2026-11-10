# AWS integration handoff — deliberately not deployed

**Project:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

This repository performs **no AWS calls**. The student will design and authorize AWS later.

## Proposed deployment responsibilities

- **Application container**: package the current Gradio app, add production authentication, persistent database, concurrency and task queue; deploy to a reviewed AWS container hosting service.
- **Storage**: S3 for raw documents, document vectors, analytical outputs, checksums and provenance; encryption and least privilege required.
- **Retrieval**: decide Bedrock Knowledge Bases plus an appropriate vector store. Replace local retrieval backend behind a documented interface.
- **Generation**: choose a currently available Bedrock model and wire a real invocation; do not hardcode a discontinued model name.
- **HPC**: ParallelCluster provisions Slurm head/compute resources, with MPI runtime matching the compile environment. Job submission must use private networking, queues or a controlled authenticated service, and return actual Slurm IDs.
- **Result contract**: preserve `document_ids`, source metadata, input dataset hash, job ID, node names, rank allocation, correctness check, wall times, result output path and error state.
- **App behavior**: use non-blocking job lifecycle (queued/running/completed/failed); never launch `mpiexec` directly from an internet-facing web callback.

## What to validate before creating anything

1. Region-specific model, vector store and EC2 availability.
2. Full budget estimate and account quota.
3. IAM/VPC/subnet networking and appropriate security groups.
4. Data access permissions and provider terms for source reports.
5. Resource teardown and secret-management plan.
6. Small end-to-end test: one document, one verified RAG response, one Slurm MPI job, one S3 result.

No subnets, credentials, cluster names or successful deployments are implied by this document.
