"""MFQ native-format definitions.

- ``nint`` -- Neuron-anchored INT codec (MFQ's core weight format with two-level neuron-anchored scales).
- ``nvq`` -- Neuron-anchored E8/D4/JSC lattice VQ (ultra-low-precision 2/3-bit formats).
- ``nvq1_l`` -- Neuron-anchored 8-D ternary lattice VQ (1.x-bit format).
- ``nvq1_s`` -- 512-entry ternary VQ using uint32/gs24 (about 1.34 bpw).
- ``npq0_l`` -- About 1.00 bpw using gs24, PQ3+4, and per-neuron anchors.
- ``npq0_s`` -- About 0.84 bpw using gs24 and state-conditioned PQ3+3.
- ``nepq`` -- Cross-expert shared table pools storing one uint8 bank ID per four gs24 groups.
- ``ternary`` -- Neuron-anchored scalar ternary reference format with five trits per byte.
- ``scheme`` -- Precision schemes describing weight/activation precision for every tensor (for example MFQ-W4.51).
- ``header`` -- MFQ file headers and tensor-level metadata.
- ``io`` -- MFQ file serialization and deserialization.
"""
