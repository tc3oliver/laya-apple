"""Exact block-local attention for the local (sliding-window) encoder layers of the BC1S body.

`_windowed` is carried over from research/phase-0-feasibility/scripts/ane_model.py
(`ConvAttention._windowed`, Phase -1 H2) without change to its computation. Instead of the
Phase -1 copy of the whole body, it is applied on top of the production
`laya_apple.conversion.bc1s.ConvBody`: only the attention modules of the local layers are
wrapped, and they reuse the production module's own qkv/out convolutions and RoPE tables. Every
other op (global layers, norms, MLPs, the decision head, the scorer) is the production graph.

Rewrite: queries are processed in blocks of `block` tokens; each block only visits the key span
[start - radius, end + radius), clipped to [0, L), with the host's additive local mask sliced
to that span. For valid query positions this is the dense masked computation restricted to the
keys it can see, so it is exact up to floating-point summation order. Padded query positions
(which see every valid key in the dense graph) see only their span here; their states are never
read, because padded keys are masked in every later layer and the CLS and marker positions are
always valid tokens.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WindowedAttention(nn.Module):
    def __init__(self, dense, *, length: int, radius: int, block: int):
        super().__init__()
        if not dense.rope:
            raise ValueError("windowed attention applies only to the encoder's RoPE attention")
        self.dense = dense  # production ConvAttention: qkv, out, cos, sin, rotate
        self.length, self.radius, self.block = length, radius, block

    def _windowed(self, qi, ki, vi, mask):
        out = []
        n, blk, r = self.length, self.block, self.radius
        for start in range(0, n, blk):
            end = min(n, start + blk)
            k0, k1 = max(0, start - r), min(n, end + r)
            q = qi[..., start:end]
            k, v = ki[..., k0:k1], vi[..., k0:k1]
            m = mask[:, k0:k1, :, start:end]
            scores = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3)) * (self.dense.dim**-0.5)
            probabilities = F.softmax(scores + m, dim=1)
            out.append(torch.einsum("bkhq,bchk->bchq", probabilities, v))
        return torch.cat(out, dim=3)

    def forward(self, x, mask):
        d = self.dense
        q, k, v = d.qkv(x).chunk(3, dim=1)
        output = []
        for qi, ki, vi in zip(q.split(d.dim, dim=1), k.split(d.dim, dim=1), v.split(d.dim, dim=1)):
            qi, ki = d.rotate(qi), d.rotate(ki)
            output.append(self._windowed(qi, ki, vi, mask))
        return d.out(torch.cat(output, dim=1))


def windowed_body(body, *, length: int, radius: int, block: int):
    """Wrap the local layers' attention of a production ConvBody in place; return (body, count)."""
    n = 0
    for layer in body.layers:
        if layer.kind != "full_attention":  # the production body's own local/global test
            layer.attn = WindowedAttention(layer.attn, length=length, radius=radius, block=block)
            n += 1
    return body, n


def dense_score_fraction(length: int, radius: int, block: int) -> float:
    """Share of the dense L x L score entries the rewrite still computes (per local layer and head)."""
    total = 0
    for start in range(0, length, block):
        end = min(length, start + block)
        total += (end - start) * (min(length, end + radius) - max(0, start - radius))
    return total / (length * length)
