# MFQ development rules

MFQ is organized around reusable inference capabilities, not checkpoint
families. New code must preserve that boundary.

## Rule zero: reusable code must not be architecture-bound

Anything that can be shared by more than one model architecture **must not**
be implemented in `models/<architecture>/` or hidden behind an
architecture-named entry point.

- Backend-neutral contracts, policies, schemas, and canonical names belong in
  `cpp_runtime/core/` or `cpp_runtime/server/`.
- Backend-wide lifecycle code, scheduling, sampling, cache policy, batching,
  metrics, and dispatch belong in `backends/<backend>/runtime/`.
- Reusable mathematical operations and packed kernels belong in
  `backends/<backend>/ops/` and `backends/<backend>/kernels/`.
- An architecture directory may contain only graph/config adaptation, source
  tensor-name import mapping, genuinely architecture-specific mathematics,
  and thin adapters to the shared runtime and operators.

The second architecture that needs an existing behavior is a mandatory
extraction point: move the behavior to the appropriate shared layer before
adding the new adapter. Do not copy, rename, or lightly modify an existing
model loop. A model name in a reusable state machine, sampler, cache
lifecycle, metric type, precision dispatcher, multimodal pipeline, MTP loop,
or serving policy is an architectural defect.

An exception is allowed only when checkpoint semantics genuinely differ. The
implementation must document that invariant next to the code and include a
test that would fail if the special path were replaced by the shared one.
Performance preference, deadline pressure, and “only one model uses it today”
are not exceptions.

## Canonical internal representation

Raw Hugging Face, GGUF, or historical MFQ tensor names may appear only at the
import/compatibility boundary. Conversion maps them once into canonical MFQ
names. Runtime operators consume canonical roles such as attention
projections, MLP gate/up/down, mHC, predictors, and multimodal components; they
must not branch on a source repository's spelling.

Compatibility readers must be isolated under an explicit `compat/` or
`legacy/` boundary. New writers emit only the canonical schema. Do not add a
silent alias merely to avoid fixing an importer.

## Backend parity

CUDA and Metal may use different kernels, but they share the same model graph,
feature capability, sampling semantics, component names, and public behavior.
Architecture support is registered from components present in the model, not
from scattered per-model booleans. Optional Vision and MTP components default
to enabled when both the graph and weights provide them; missing weights or an
explicit user override disables them.

## Change checklist

Before submitting a runtime or quantization change:

1. Identify which layer owns the behavior: core, backend runtime, reusable
   operator/kernel, importer, or architecture adapter.
2. Search all architectures and both backends for an equivalent implementation.
3. Extract reusable behavior before extending it; keep model adapters thin.
4. Add a behavior test and, for critical ownership rules, a source-boundary
   test under `tests/`.
5. Run the focused native tests, the Python suite, and the full CTest suite when
   the change touches shared runtime behavior.
6. Do not publish generated files, local paths, credentials, model weights, or
   disposable benchmark artifacts.

Code review must reject architecture-local duplication even when it works for
the first model. The maintenance cost is paid by every model added afterward.
