# Offline proof benchmarks

These scripts measure Pearl proof generation and verify saved certificates
locally. They do not contact a gateway or node, submit work, or change chain
state. Keep `bench_prover.py` and `verify_certificates.py` together: the verifier
imports shared metadata helpers from the prover script.

Proof latency and memory use depend on the witness shape, CPU, compiler target,
thread count, and whether the circuit cache is warm. A single result should not
be treated as a general performance guarantee. When evaluating a change, compare
equivalent fixtures under the same CPU and memory limits, and independently
verify the resulting certificates with a compatible, unchanged verifier.

## Build

From an isolated Python environment in `py-pearl-mining/`, build with a Rust
target supported by the host that will run the extension. For a build used only
on the same host:

```sh
RUSTFLAGS='-C target-cpu=native' maturin develop --release
```

For a portable build, choose an explicit baseline supported by every target
CPU. A model name alone is not enough to establish support for a compiler
target; check the actual CPU feature set. The benchmark records the loaded
extension's path and SHA-256 so runs can be tied to the binary that produced
them. The metadata self-tests do not require an installed Pearl extension:

```sh
python benchmarks/bench_prover.py --self-test-metadata
python benchmarks/verify_certificates.py --self-test-metadata
```

## Measure proof generation

Use an authorized local `PlainProof` fixture and matching header metadata.
The raw-fixture example below measures proof generation for that witness.
`--input-fixture` is an alternative when the fixture already contains its
header, shape, and proof. Use a fresh path for `--output-fixture` so saved
certificates can be verified separately.

```sh
cd py-pearl-mining
WORK_DIR="$(mktemp -d)"
RAYON_NUM_THREADS=6 RUST_LOG=warn \
python benchmarks/bench_prover.py \
  --raw-base64-fixture /path/to/plain-proof.b64 \
  --header-version "$HEADER_VERSION" \
  --header-prev-block-hex "$PREV_BLOCK_HEX" \
  --header-merkle-root-hex "$MERKLE_ROOT_HEX" \
  --header-timestamp "$TIMESTAMP" \
  --nbits "$NBITS" \
  --rows 16 --cols 16 \
  --repetitions 3 \
  --output-fixture "$WORK_DIR/certificates.jsonl" \
  >"$WORK_DIR/prover.jsonl" 2>"$WORK_DIR/prover.stderr"
```

Set `RAYON_NUM_THREADS` before importing `pearl_mining`; the extension creates
its global Rayon thread pool during import. For optional stage-level diagnostics,
set `PEARL_PROVER_PROFILE=1` and
`RUST_LOG='info,plonky2::util::timing=debug'`. Profiling can perturb timings,
so keep it off for primary latency comparisons.

## Verify saved certificates

Run the verifier offline against the generated JSONL, preferably using an
independently built compatible library:

```sh
python benchmarks/verify_certificates.py "$WORK_DIR/certificates.jsonl" \
  >"$WORK_DIR/verification.jsonl" 2>"$WORK_DIR/verification.stderr"
```

Each record reports the verifier extension hash and positive and
altered-header-negative checks. Verification of a saved certificate does not
establish network submission or block acceptance.

## Interpret results

- `timings.prove` is proof-generation latency; verification and mining/search
  costs are separate. A supplied fixture avoids measuring fresh mining search.
- The first repetition is marked `cold` after clearing the in-process circuit
  cache; later repetitions are `warm` in the same process. This does not clear
  filesystem or external caches.
- `memory.ru_maxrss_bytes` is a process high-water mark, not a per-proof peak.
  `current_rss_bytes` is a point-in-time sample. Check cgroup `memory.peak`
  separately when a cgroup-level peak is needed.
- Synthetic fixtures can be useful for reproducibility but do not establish
  latency or memory use for live miner witnesses. Record the fixture shape and
  limits without publishing private witness contents.

Run one benchmark at a time under verified host-level CPU and memory limits.
Do not infer that a container memory flag is enforced without checking the
process's actual cgroup limits. These commands remain offline and never submit
certificates to a chain.
