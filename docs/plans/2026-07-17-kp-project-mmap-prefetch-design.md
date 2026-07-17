# KP Project mmap Prefetch Design

## Goal

Reduce end-to-end `kp project` time for large all-k Hamiltonians while preserving
`project.gauge: auto`, the configured downfolding method, and numerical output.

The motivating MgI2 Gamma q06 case contains 61 dense complex Hamiltonians. Each k
slice is about 0.69 GiB, the full file is about 42 GiB, and linearized Lowdin builds
and factors a 6660 by 6660 high-energy block for every k.

## Evidence

- The target bigmem002 node exposes 128 physical CPUs: four sockets, 32 cores per
  socket, one hardware thread per core, and four NUMA nodes. Hyperthreading is not
  exposed.
- The current default uses four workers and eight MKL threads per worker, so the
  formal projection can use at most 32 of the 128 physical cores. This matches the
  observed 20 to 25 percent CPU utilization during the 61-k run.
- Initial algorithm probes were accidentally run on login002, which exposes 64
  physical CPUs (two sockets, 32 cores per socket). Those probes establish matrix
  dimensions and candidate optimization ideas, but their timings are not valid
  bigmem002 performance measurements.
- On login002, a gauge-auto single-k A/B probe reduced runtime from 12.78 s to
  10.17 s by making one sequential 0.69 GiB copy before random block extraction.
  This is a hypothesis to re-test on bigmem002, not a production benchmark result.
- The q06 downfold uses HPP `(148, 148)`, HPH `(148, 6660)`, and HHH `(6660, 6660)`.
  Login002 phase timings indicate that block assembly and dense LU dominate; the
  phase breakdown must be repeated on bigmem002 before selecting thread defaults.

## Selected Approach

Keep automatic gauge resolution unchanged. Immediately after selecting the spin
block for a k point, detect a large mmap-backed/non-owning two-dimensional matrix
and copy it sequentially into C-contiguous local memory before projector block
assembly. Small or already-owned arrays retain the existing zero-copy path.

The behavior is automatic and does not require users to provide anchors or change
the YAML. A project-level escape hatch may disable prefetch for constrained-memory
systems, but the default is `auto`.

Before implementation, repeat the prefetch hypothesis and phase timing through SSH
on bigmem002. After prefetch is implemented, benchmark the exact gauge-auto q06 path
there with 4x8, 4x16, 8x8, and 16x4 worker/thread layouts. Do not change the worker
default solely to increase nominal CPU utilization: select a new default only if
end-to-end throughput improves without numerical changes.

## Data Flow

1. Resolve auto gauge and certify the production basis exactly as today.
2. Each worker maps the Hamiltonian file once.
3. For each assigned k, select the requested spin block.
4. In auto-prefetch mode, sequentially copy a sufficiently large mmap-backed k
   slice to owned C-contiguous memory.
5. Build projector blocks and run the configured Lowdin/Schur downfold unchanged.
6. Return Heff, wavefunctions, diagnostics, and plots through the existing path.

## Reporting

`kp project` should always report:

- requested and effective worker counts;
- BLAS threads per worker;
- process-visible CPU count;
- prefetch policy and per-k slice size when prefetch is active.

This prevents scheduler allocation, visible cpuset, and actual numerical-library
parallelism from being conflated.

## Correctness and Fallback

- Prefetch changes storage ownership only; matrix values and basis order are
  unchanged.
- Auto gauge remains mandatory for the release example and is not replaced with
  persisted manual anchors.
- Allocation failure in explicit `on` mode is an error. In `auto` mode, the code
  may retain the mmap path and report the fallback rather than changing physics.
- Existing global thread environment variables are not set.

## Tests and Acceptance

- Unit tests prove that auto mode copies a large mmap-backed/non-owning k slice,
  leaves small/owned arrays unchanged, and honors an explicit off setting.
- CLI tests prove that effective workers, BLAS threads, visible CPUs, and prefetch
  policy are reported.
- Numerical regression compares gauge report, Heff, eigenvalues, and wavefunctions
  between direct mmap and prefetch paths.
- The MgI2 q06 benchmark uses `gauge:auto` throughout and reports Auto-gauge time,
  formal projection time, and end-to-end 61-k time separately. The report records
  `hostname`, CPU model, socket/core/thread topology, and process affinity so results
  from login and compute nodes cannot be conflated.
