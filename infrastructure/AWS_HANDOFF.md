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

## Student-owned execution order (not run by this release)

1. Choose an AWS region and record a budget ceiling, EC2 and service quotas,
   data residency requirements, and the actual prices used for estimates.
2. Design a private VPC/subnet layout. Give the web service, S3 objects,
   Slurm head node, and compute nodes separate least-privilege roles. Keep
   job submission off the public web callback and avoid stored access keys.
3. Build and smoke-test the app container locally. Add authentication,
   multi-user storage isolation, durable job state, and a queue before hosting.
   The current JSON store and synchronous local callback are not production
   infrastructure.
4. Define versioned S3 prefixes for permitted raw reports, source metadata,
   vector NPZ input, candidate results, verified results, and logs. Retain
   content SHA-256 and the vector fingerprint in each submitted job record.
5. If Bedrock is selected, verify regional model availability and invoke a
   real model using retrieved, licensed source excerpts. Record model ID,
   cost, answer, citation review, and failure handling. Keep the default
   evidence-only path operational.
6. Provision ParallelCluster only after budget approval. Use Slurm to submit
   the existing MPI operation with the same Python and MPI ABI on compute
   nodes. Start with one node and two ranks, then one node/four ranks, then
   a controlled two-node case on the **same input fingerprint**.
7. Capture the actual Slurm job ID, assigned hostnames, rank IDs, row ranges,
   stdout/stderr, wall time, peak memory, output checksum, and a separately
   computed Sequential correctness result. A queue acceptance is not a
   completed or correct job.
8. Publish a result to the app only after correctness succeeds. Store queued,
   running, completed, failed, timed-out, and cancelled states durably;
   restrict logs and documents to the requesting user.
9. Calculate cost from actual region, instance rate, node count, billed
   duration, S3 storage/requests, relevant transfers, app hosting, and model
   inference. Distinguish measured bill lines from estimates. Compare
   Sequential and MPI cost per validated work unit, not only elapsed time.
10. Stop/delete idle resources after evidence capture and preserve anonymised
    experiment metadata for the academic report. Update the professor
    checklist only with the observed cloud values.

`infrastructure/slurm/` contains review templates for similarity and
clustering. They require real cluster paths, environment modules, queue
names, storage, and permissions chosen by the student; none are prefilled as
if a deployment exists.

## Evidence to bring back for the final submission

Record the actual AWS region, service names, EC2 types and node counts,
matching Python/MPI versions, S3 input and output object keys, input/output
SHA-256 hashes, Slurm submission command and job ID, and rank-to-host log.
Capture dated screenshots of the secured app, completed Slurm job, S3 objects,
and billing view **only after they exist**. Keep stdout/stderr, Sequential
comparison, wall time, throughput, speedup, efficiency, memory, and the actual
billed duration and rates together. The S3 contract must distinguish raw
reports, source metadata, vector input, candidate output, verified output and
logs, with explicit access and retention rules. After collecting evidence,
stop the cluster and any idle hosted service, then check for remaining charges.
No such cloud artifacts are claimed in this local release.
