# NINTv2 and NAQ-imatrix Development Roadmap

NINTv2 and NAQ-imatrix are foundational explorations into using the structure
and behavior of neural networks to push quantization closer to the
rate–distortion frontier. They are likely to become a major development and
maintenance focus for MFQ over the coming period.

## NINTv2: precision as a per-neuron resource

NINTv2 began with a simple question: how can we assign precision to individual
neurons?

Research into per-neuron precision is not new, but efficient implementations
remain uncommon. The NINTv1 abstraction can already express this idea with
reasonable efficiency. Important Neurons (IN) can split neurons of different
importance into several Sub-FFNs with different NINT precisions, after which a
heterogeneous dispatcher executes those branches in parallel.

That approach is not especially slow, but it is inelegant and its number of
possible combinations can grow rapidly. Applying the same construction to an
MoE model could produce thousands of logical FFNs that are activated sparsely.
Careful engineering could still make such a system practical, but the
complexity is unnecessary when a simpler representation is available.

NINTv2 turns both primary value-code width, `q`, and subgroup-metadata width,
`k`, into per-neuron resources without turning every possible assignment into
a new format. Its on-disk representation adds a three-bit `q` selector and a
two-bit `k` selector per neuron, then stores rows in compact width cohorts. The
five descriptor bits cost only about 0.001 bpw for a typical 4K--5K-wide
neuron. Subgroup metadata is expanded once into the existing compact runtime
table, while the primary values remain packed and are addressed through
per-row bit offsets. One dynamic packed kernel can therefore execute different
`q` widths together without persistent FP16 unpacking or a collection of
precision-specific Sub-FFNs.

The result is one tensor, one dispatch contract and one physical format even
when every neuron receives a different `(q, k)` choice. The surrounding graph
does not need to understand that internal allocation. Current Metal
measurements on a 4096 x 5120 matrix put the heterogeneous path within roughly
5% of fixed NINT4 at M=1, and at parity in the sampled M=2--16 range, without a
configuration-specific fast path. More importantly, the representation avoids
the combinatorial growth in logical tensors and kernel dispatches that made
fine-grained precision allocation awkward in NINTv1.

NINTv2 also fits naturally inside heterogeneous containers such as NINTM.
NINTM registers it simply as `NINTv2`; the per-neuron precision map remains
inside the tensor payload. Experts do not need to be split or regrouped by
their internal assignments, and two NINTv2 tensors may use entirely different
precision distributions while presenting the same format identity to the
container and runtime.

Conceptually, the fixed NINTv1 profiles become a small set of presets inside
the NINTv2 search space. A conventional NINT4 profile is the point obtained by
assigning the same `q` and `k` to every neuron; other familiar NINTv1 profiles
are analogous uniform points. They remain useful as simple, robust presets and
fixed-kernel baselines, but they no longer define the boundary of the
representable precision space. NINTv2 is better described by an aggregate
`xbpw` budget and the allocation of that budget across neurons than by a
single nominal bit width.

NINTv2 is not necessarily the endpoint, but the next useful degree of freedom
is less obvious. NINTv3 might allow each subgroup to choose its own metadata
precision, or allow each neuron to use a different group size. Both directions
are substantially harder to encode and execute efficiently: they introduce
more irregular addressing and can fragment the kernel's group geometry. They
may also face diminishing returns. Input-channel imatrix weighting can already
capture much of the useful allocation structure within one neuron; the larger
first-order question is how much total budget that neuron deserves. A future
NINTv3 should therefore be justified by measured gains that exceed this added
format, calibration and kernel complexity, rather than by a larger search
space alone.

Every additional degree of freedom also requires calibration methodology and
calibration scale to match. Otherwise, the enlarged search space can easily
overfit. This raises the demands placed on calibration, which is increasingly
becoming the bottleneck. NINTv2 therefore does not simply supersede NINTv1: it
offers a higher attainable ceiling and, with sound calibration, should almost
always outperform NINTv1. Without that calibration, the additional freedom is
not automatically beneficial.

## NAQ-imatrix: importance in the context of a neural network

NAQ-imatrix extends the same neural-network-aware principle to calibration. A
conventional imatrix considers input-side moments. This is useful for estimating
the relative importance of input channels, but it cannot express the importance
of the neurons themselves, even though neurons within a tensor are not equally
important.

NAQ also accounts for activation functions. Nonlinear projections are becoming
increasingly common in modern neural networks, particularly with Gated Linear
Attention and the growing family of gated sparse-attention mechanisms. Layers
with Sigmoid or SiLU cannot be evaluated only by the Euclidean energy of their
pre-activation outputs. A neuron that consistently produces a large negative
pre-activation may appear energetic under that metric, yet its post-activation
output remains close to zero and its quantization error has little downstream
effect.

NAQ-imatrix therefore records both input-channel moments and output-neuron
importance. For nonlinear components, it also incorporates the actual
activation and downstream sensitivity. It treats a matrix according to its role
inside the neural network rather than as an isolated weight array, providing
signals for both neuron-level precision allocation and channel-level allocation
within each neuron.

## Repositioning Important Neurons

The role of IN is being reconsidered because NINTv2 absorbs much of its original
weight-precision function. IN is now primarily aimed at quantization-aware
training and Mixed Activation Precision (MAP).

Some neurons consistently exhibit large activation spikes, making low-precision
activation quantization disproportionately destructive. Many other neurons may
have much better-behaved activation distributions and can be quantized safely,
enabling native low-precision dot-product acceleration for Prefill on NVIDIA
hardware. In this setting, IN identifies and stores the neurons whose
activations should remain at higher precision. A separate execution path is
appropriate here because it represents a genuinely different activation
contract rather than redundant weight-precision partitioning. Supporting
different weight precisions remains useful, but it becomes a secondary property
of IN rather than its primary purpose.

## Native-QAT-aware SQ formats

MXFP4-SQ was created from a related observation. As more models are released
with native MXFP4, MXFP8, or FP8 QAT weights, treating BF16 weights as the
universal source of truth increasingly becomes a false premise. Conditioning a
format on the model's actual source precision creates useful rate–distortion
and representation opportunities.

MXFP4-SQ2 and MXFP4-SQ3 were designed for native MXFP4 weights. Their decoded
values never leave the set of values representable by MXFP4. In our tests to
date, they outperform equal-size VQ formats and larger conventional SQ formats
on most real native-MXFP4 weights, while remaining fast to quantize and unpack.

They also address a practical systems question: can a custom high-fidelity
quantization format benefit from native low-precision hardware acceleration?
Every value produced by MXFP4-SQ is a legal MXFP4 value, so the packed
representation can be decoded directly into an MXFP4 compute path without
materializing a fully pre-unpacked copy of the weights. This requirement is one
reason the design uses scalar quantization. In practice, its rate–distortion
efficiency has already proved sufficient for this role, exceeding our initial
expectations.

The next planned formats are MXFP8-SQ4/5/6 and FP8-SQ4/5/6. Their goal is to
provide high-fidelity compression for native FP8 QAT weights while retaining
access to W8A8 acceleration.
