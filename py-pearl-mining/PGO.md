# Optional profile-guided builds

Profile-guided optimization (PGO) is a build-time option, separate from the
source-level prover optimizations. Compare baseline, source-only, and
source-plus-PGO builds independently; a combined gain is not a source-only gain.

Use the same source revision, lockfile, Rust toolchain, target triple, linker,
CPU target, feature flags, and release settings for instrumentation and the
final build. Choose a representative offline proof-generation workload with
local inputs. A hash microbenchmark alone does not train the whole prover.

## Instrument and train

From this directory, with the intended Rust toolchain selected:

```sh
set -eu
umask 077
PGO_ROOT="$(mktemp -d)"
mkdir -p "$PGO_ROOT/raw"

# Use an explicit compatible CPU target when build and run machines differ.
COMMON_RUSTFLAGS="-C target-cpu=native"
RUSTFLAGS="$COMMON_RUSTFLAGS -C profile-generate=$PGO_ROOT/raw" \
  cargo build --release --locked --target-dir "$PGO_ROOT/target-generate"
```

Run your offline training workload with the instrumented library from
`$PGO_ROOT/target-generate/release`, setting
`LLVM_PROFILE_FILE="$PGO_ROOT/raw/%p-%m.profraw"` in its environment. Verify
which native library the workload actually loads: `cargo build` does not
install the extension into a Python environment. Configure your packaging or
test harness accordingly, without replacing a live application's library.

Instrumentation can greatly increase time and memory use. Apply a timeout and
resource limits, reduce training concurrency if necessary, and let successful
training processes exit normally to flush their profiles. Training timings
are not production performance measurements.

## Merge and rebuild

Use `llvm-profdata` from the same compiler toolchain, available through Rust's
`llvm-tools-preview` component. Set `LLVM_PROFDATA` to that executable, then:

```sh
: "${LLVM_PROFDATA:?Set LLVM_PROFDATA to the matching llvm-profdata executable}"
set -- "$PGO_ROOT"/raw/*.profraw
test -e "$1"
"$LLVM_PROFDATA" merge --sparse -o "$PGO_ROOT/training.profdata" "$@"
"$LLVM_PROFDATA" show "$PGO_ROOT/training.profdata"

RUSTFLAGS="$COMMON_RUSTFLAGS -C profile-use=$PGO_ROOT/training.profdata" \
  cargo build --release --locked --target-dir "$PGO_ROOT/target-use"
```

Keep target directories separate. Investigate missing or mismatched profile
warnings rather than assuming the resulting build is fully profile-guided.
Package the final profile-use artifact, never the instrumented training build.
If packaging recompiles the extension, it must retain the profile-use flags
and all matching build settings.

## Validate and measure

Run the normal tests and full proof generation with the final artifact. An
unchanged compatible verifier must accept valid certificates and reject
controlled altered-header or altered-proof cases. Record artifact identities,
fixture shapes, worker counts, cold/warm timings, and peak memory privately.
Use matched repeated runs to distinguish a real gain from runtime variation.

Keep generated profiles and raw logs outside the repository: they can contain
paths, symbols, and workload details. Do not commit or publish them. Retrain
and revalidate when the source, compiler, build settings, or workload changes.
