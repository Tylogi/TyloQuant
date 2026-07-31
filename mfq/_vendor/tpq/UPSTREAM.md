# Vendored TyloQuant PQ runtime

This directory contains the production dependency closure used by MFQ's CCCP
runtime integration. It was synchronized from the collaborator-maintained
snapshot at `references/tpq2` on 2026-07-29.

The vendored package keeps the runtime import name `tpq` because its CUDA
extension and internal relative imports use that package boundary. MFQ loads
an explicitly selected external TPQ tree first, then an installed `tpq`
package, and finally this vendored copy.

Production Python modules, static architecture configurations, native CPU/CUDA
sources, and their module documentation are included here. Generated archives,
caches, and bytecode remain outside the MFQ package.
