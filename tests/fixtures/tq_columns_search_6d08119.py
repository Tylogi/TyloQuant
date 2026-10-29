"""Upstream 6d08119 search oracle; tests inject the same production encoder."""
import torch
from mfq.formats.tq import fixed_codebook

@torch.inference_mode()
def prepare_columns(weight):
    """One precision-independent scale vector shared by every row/candidate."""
    w = weight.float()
    rms = torch.tensor(fixed_codebook(), device=w.device).float().square().mean().sqrt()
    columns = (w.square().mean(0).sqrt() / w.square().mean().sqrt().clamp_min(2**-24)).clamp_min(.01).half()
    for _ in range(2):
        anchor = ((w / columns.float()).square().mean(-1).sqrt() / rms).clamp_min(2**-24).half()
        columns = ((w / anchor.float()[:, None]).square().mean(0).sqrt() / rms).clamp_min(.01).half()
    return columns


@torch.inference_mode()
def fit_columns(weight, k, h, columns, *, fast=False, repair_scales=True, allow_clipping=False):
    """Fit fixed-column TQ rows with full source-range coverage by default."""
    w = weight.float()
    rows, width = w.shape
    table = torch.tensor(fixed_codebook(), device=w.device).float()
    rms = table.square().mean().sqrt()
    row_rms = (w / columns.float()).square().mean(-1).sqrt() / rms
    anchor = row_rms.clamp_min(2**-24).half()
    floor = None
    if not allow_clipping:
        bound = (w.abs() / (torch.where(w >= 0, table.max(), -table.min()) *
                           columns.float()[None, :])).amax(-1)
        floor = bound.half()
        floor = torch.where(floor.float() < bound,
            torch.nextafter(floor, torch.full_like(floor, torch.inf)), floor)
    factors = ((.65, .7, .75, .8, .85, .9, .95, 1., 1.05, 1.1, 1.2, 1.3)
               if k == 1 else (1., .9, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.))

    def trial(target, a, lower=None):
        if lower is not None:
            a = torch.maximum(a, lower)
        scale = a.float()[:, None] * columns.float()[None, :]
        metric = scale * h.sqrt()[None, :]
        metric /= metric.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
        y, state, _ = encode((target / scale).reshape(-1, 256).contiguous(),
                             metric.reshape(-1, 256).contiguous(), k)
        basis = y.reshape_as(target) * columns.float()[None, :]
        proposal = ((target * basis * h).sum(-1) /
                    (basis.square() * h).sum(-1).clamp_min(1e-30))
        proposed = proposal.half()
        # Invalid or negative LS must not turn a real row into near-zero weights.
        # Also reject anchors that would overflow FP16 encoder inputs on reuse.
        minimum = a.float() * (target / scale).abs().amax(-1) / 65504.
        valid = torch.isfinite(proposed) & (proposed.float() > minimum)
        proposed = torch.where(valid, proposed, a)
        if lower is not None:
            proposed = torch.maximum(proposed, lower)
        old = ((target - (basis * a.float()[:, None]).half().float()).square() * h).sum(-1)
        new = ((target - (basis * proposed.float()[:, None]).half().float()).square() * h).sum(-1)
        return state.reshape_as(target), torch.where(new < old, proposed, a), torch.minimum(new, old)

    ids = torch.linspace(0, rows - 1, min(rows, 16 if fast else 64), device=w.device).long()
    factor_table = torch.tensor(factors, device=w.device)
    sample_a = (anchor[ids].float()[None, :] * factor_table[:, None]).half().flatten()
    _, _, loss = trial(w[ids].repeat(len(factors), 1).contiguous(), sample_a,
                       None if floor is None else floor[ids].repeat(len(factors)))
    order = loss.reshape(len(factors), len(ids)).sum(-1).argsort()
    best = torch.full((rows,), torch.inf, device=w.device)
    states = torch.empty_like(w, dtype=torch.int16)
    anchors = anchor.clone()
    for index in order[:1 if fast else 2]:
        state, a, loss = trial(w, (anchor.float() * factor_table[index]).half().clamp_min(2**-24), floor)
        take = loss < best
        best[take], anchors[take], states[take] = loss[take], a[take], state[take]
    if not fast:
        state, a, loss = trial(w, anchors.clone(), floor)
        take = loss < best
        best[take], anchors[take], states[take] = loss[take], a[take], state[take]
    if repair_scales:
        from .tq_scale_repair import propose_scales
        proposals = propose_scales(w, h, columns, anchors, best)
        ids = torch.nonzero(proposals > anchors).flatten()
        if len(ids):
            state, a, loss = trial(w[ids].contiguous(), proposals[ids],
                                  None if floor is None else floor[ids])
            take = loss < best[ids]
            selected = ids[take]
            anchors[selected], states[selected] = a[take], state[take]
    anchors.masked_fill_(row_rms == 0, 0)
    return states, anchors
