"""Gen 5 Galcon-Native Edge-Gated Transformer with Residual Pairwise Edge Track & WDL Value Head.

Self-contained inside ``galcon/`` (zero modifications to ``orbit_wars_bot/``) and drop-in
compatible with ``GalconTransformerAgent``, ``GalconSearchTransformerAgent``,
``EnsembleGalconSearchAgent``, ``galcon/train.py``, and ``galcon/gbots_client.py``.

Key architectural upgrades over Gen 4 (``1.29M`` static-edge ``GalconPlanetTransformer``):
1. **Edge-Conditioned Value Message Passing + Residual Edge Track (``edge_hidden=24``):**
   Every transformer layer passes edge-conditioned tactical messages into planet embeddings
   *and* refines the ``(B, N, N, edge_hidden)`` pairwise edge tensor using updated ``(h_i, h_j)``
   states, enabling explicit 2-hop / pincer / rescue-chain coordination.
2. **Pre-Norm SwiGLU Feed-Forward Blocks (``d_model=192, n_heads=6, n_layers=6, dim_ff=384``):**
   Higher representation capacity per FLOP (~``2.93M`` params) with configurable residual
   dropout (zero inference cost at ``.eval()``).
3. **Multi-Query Attention Pooling + Stockfish/Lc0-Style WDL Value Head:**
   4 learned global query tokens attend over frontline/backline planets to predict
   ``(P_loss, P_draw, P_win)`` via 3-class softmax. Expected value
   ``v = P_win - P_loss - contempt * P_draw`` separates sterile draws from sharp tactical fights.
4. **Zero-Waste Native Multi-Send Architecture:**
   Drops the unused Gen 2/3 legacy pairwise head and feeds the layer-6 refined edge tensor
   directly into ``predict_multi_send``.
"""

from __future__ import annotations

import copy
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, hidden_dim * 2)
        self.w_down = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class EdgeGatedTransformerBlock(nn.Module):
    """Pre-Norm Graph Transformer block with edge-biased attention, edge-value messages, SwiGLU, and edge updates."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        edge_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.edge_hidden = edge_hidden
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.edge_bias = nn.Linear(edge_hidden, n_heads)
        self.edge_val = nn.Linear(edge_hidden, edge_hidden)
        self.edge_msg_out = nn.Linear(n_heads * edge_hidden, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _SwiGLU(d_model, dim_ff)

        # Residual pairwise edge update: e_ij <- e_ij + MLP(LayerNorm(e_ij) + W_s(h_i) + W_d(h_j))
        self.edge_norm = nn.LayerNorm(edge_hidden)
        self.edge_src = nn.Linear(d_model, edge_hidden, bias=False)
        self.edge_dst = nn.Linear(d_model, edge_hidden, bias=False)
        self.edge_upd = nn.Sequential(
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, D), e: (B, N, N, De), pad_mask: (B, N) bool (True=valid)
        B, N, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        xn = self.norm1(x)
        qkv = self.qkv(xn).view(B, N, 3, H, Dh)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, H, N, Dh)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        eb = self.edge_bias(e).permute(0, 3, 1, 2)  # (B, H, N, N)
        logits = logits + eb
        logits = logits.masked_fill(~pad_mask[:, None, None, :], -1e9)
        attn = F.softmax(logits, dim=-1)

        node_ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        ev = self.edge_val(e)  # (B, N, N, De)
        # AVX2-backed batched GEMM: (B*N, H, N) @ (B*N, N, De) -> (B*N, H, De)
        attn_bn = attn.transpose(1, 2).reshape(B * N, H, N)
        ev_bn = ev.reshape(B * N, N, self.edge_hidden)
        edge_ctx = torch.bmm(attn_bn, ev_bn).reshape(B, N, H * self.edge_hidden)
        x = x + self.drop(self.out_proj(node_ctx) + self.edge_msg_out(edge_ctx))

        xn2 = self.norm2(x)
        x = x + self.drop(self.ffn(xn2))

        e_in = self.edge_norm(e) + self.edge_src(xn2).unsqueeze(2) + self.edge_dst(xn2).unsqueeze(1)
        e = e + self.drop(self.edge_upd(e_in))
        return x, e


class GalconGen5Transformer(nn.Module):
    """Gen 5 Edge-Gated Galcon Transformer with Residual Edge Track, SwiGLU FFN, and WDL Value Head."""

    def __init__(
        self,
        *,
        planet_dim: int = 22,
        timeline_dim: int = 8,
        edge_dim: int = 12,
        redir_dim: int = 10,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        dim_ff: int = 384,
        edge_hidden: int = 24,
        num_val_queries: int = 4,
        feature_version: int = 2,
        contempt: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_version = int(feature_version)
        self.enable_multi_send = True
        self.contempt = float(contempt)
        self.edge_hidden = int(edge_hidden)
        self.d_model = int(d_model)
        self.dropout = float(dropout)
        self.config = {
            "model_class": "GalconGen5Transformer",
            "planet_dim": int(planet_dim),
            "timeline_dim": int(timeline_dim),
            "edge_dim": int(edge_dim),
            "redir_dim": int(redir_dim),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "dim_ff": int(dim_ff),
            "edge_hidden": int(edge_hidden),
            "num_val_queries": int(num_val_queries),
            "feature_version": int(feature_version),
            "enable_multi_send": True,
            "dropout": float(dropout),
        }

        # Multi-scale temporal conv over 20-tick exact arrival timelines (mean + max + final-tick pool -> 192d)
        self.timeline_conv = nn.Sequential(
            nn.Conv1d(timeline_dim, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.node_in = nn.Sequential(
            nn.Linear(planet_dim + 192, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.LayerNorm(edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden),
        )

        self.layers = nn.ModuleList(
            [
                EdgeGatedTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    edge_hidden=edge_hidden,
                    dropout=self.dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.final_edge_norm = nn.LayerNorm(edge_hidden)
        self.head_drop = nn.Dropout(self.dropout)

        # Native Gen 5 pairwise multi-target send head using BOTH final node embeddings and refined edge track
        ms_hid = d_model // 2
        self.ms_src_proj = nn.Linear(d_model, ms_hid)
        self.ms_dst_proj = nn.Linear(d_model, ms_hid)
        self.ms_edge_proj = nn.Linear(edge_hidden, ms_hid)
        self.ms_raw_edge_proj = nn.Linear(edge_dim, ms_hid)
        self.ms_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ms_hid, ms_hid // 2),
            nn.SiLU(),
            nn.Linear(ms_hid // 2, 2),
        )
        nn.init.constant_(self.ms_head[-1].bias[0], -2.2)
        nn.init.constant_(self.ms_head[-1].bias[1], -0.4)

        # Multi-Query Attention Pooling + 3-Class WDL Value Head: [loss, draw, win]
        self.val_queries = nn.Parameter(torch.randn(num_val_queries, d_model) * 0.04)
        self.val_key = nn.Linear(d_model, d_model, bias=False)
        self.val_val = nn.Linear(d_model, d_model, bias=False)
        val_in_dim = (num_val_queries + 1) * d_model
        self.val_wdl_head = nn.Sequential(
            nn.Linear(val_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )

        # Auxiliary t+10 future-state world-model head: [owner_neu, owner_me, owner_en, ships_log]
        self.aux_future_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 4),
        )

        # Native D4-invariant mid-flight /REDIR head
        self.redir_cur_proj = nn.Linear(d_model, ms_hid)
        self.redir_dst_proj = nn.Linear(d_model, ms_hid)
        self.redir_feat_proj = nn.Linear(redir_dim, ms_hid)
        self.redir_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ms_hid, 1),
        )

    def encode_planets_and_edges(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute contextual planet embeddings ``h (B, N, D)`` and refined edge tensor ``e (B, N, N, De)``."""
        B, N, T, Ct = timeline_feats.shape
        tl = timeline_feats.reshape(B * N, T, Ct).transpose(1, 2)
        tl_h = self.timeline_conv(tl)
        tl_pool = torch.cat([tl_h.mean(dim=-1), tl_h.amax(dim=-1), tl_h[:, :, -1]], dim=-1).view(B, N, 192)
        x = self.node_in(torch.cat([planet_feats, tl_pool], dim=-1))
        e = self.edge_in(edge_feats)
        for layer in self.layers:
            x, e = layer(x, e, pad_mask)
        return self.final_norm(x), self.final_edge_norm(e)

    def encode_planets(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return contextual planet embeddings ``h (B, N, D)`` (drop-in compatible with Gen 4)."""
        h, _ = self.encode_planets_and_edges(planet_feats, timeline_feats, edge_feats, pad_mask)
        return h

    def decode_wdl_from_hidden(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Predict 3-class WDL logits ``(B, 3)`` ordered as ``[loss, draw, win]``."""
        B, N, D = h.shape
        k = self.val_key(h)  # (B, N, D)
        v = self.val_val(h)  # (B, N, D)
        # Multi-query attention: (Q, D) x (B, N, D) -> (B, Q, N)
        attn_logits = torch.einsum("qd,bnd->bqn", self.val_queries, k) * (D ** -0.5)
        attn_logits = attn_logits.masked_fill(~pad_mask.unsqueeze(1), -1e9)
        attn_w = F.softmax(attn_logits, dim=-1)
        q_pool = torch.matmul(attn_w, v).reshape(B, -1)  # (B, Q*D)

        # Valid-planet mean summary
        pm_f = pad_mask.float().unsqueeze(-1)
        mean_pool = (h * pm_f).sum(dim=1) / pm_f.sum(dim=1).clamp(min=1.0)
        return self.val_wdl_head(self.head_drop(torch.cat([q_pool, mean_pool], dim=-1)))

    def decode_value_from_hidden(
        self,
        h: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        contempt: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute scalar value ``(B,)`` in ``[-1, 1]`` from WDL probabilities: ``P_win - P_loss - c * P_draw``."""
        wdl = F.softmax(self.decode_wdl_from_hidden(h, pad_mask), dim=-1)
        c = self.contempt if contempt is None else float(contempt)
        val = wdl[:, 2] - wdl[:, 0] - c * wdl[:, 1]
        return val.clamp(-1.0, 1.0)

    def predict_multi_send(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        *,
        edge_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict independent pairwise ``(ms_logits, ms_fracs)`` of shape ``(B, N, N)`` each."""
        eh = self.edge_in(edge_feats) if edge_hidden is None else edge_hidden
        pair = (
            self.ms_src_proj(h).unsqueeze(2)
            + self.ms_dst_proj(h).unsqueeze(1)
            + self.ms_edge_proj(eh)
            + self.ms_raw_edge_proj(edge_feats)
        )
        out = self.ms_head(self.head_drop(pair))
        return out[..., 0], torch.sigmoid(out[..., 1])

    def predict_aux_future(self, h: torch.Tensor) -> torch.Tensor:
        """Predict per-planet ``t+10`` state ``(B, N, 4)``: ``[own_neu, own_me, own_en, log_ships]``."""
        return self.aux_future_head(h)

    def predict_redir_logits(
        self,
        h: torch.Tensor,
        cur_dst_idx: torch.Tensor,
        redir_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate redirect targets ``(B, F, N)`` for flying fleets."""
        B, N, D = h.shape
        gather_idx = cur_dst_idx.clamp(0, N - 1).unsqueeze(-1).expand(-1, -1, D)
        h_cur = torch.gather(h, 1, gather_idx)
        pair = (
            self.redir_cur_proj(h_cur).unsqueeze(2)
            + self.redir_dst_proj(h).unsqueeze(1)
            + self.redir_feat_proj(redir_feats)
        )
        logits = self.redir_head(pair).squeeze(-1)
        return logits.masked_fill(~pad_mask.unsqueeze(1), -1e9)

    def forward_multi_send(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        return_aux: bool = False,
        return_hidden: bool = False,
        return_wdl: bool = False,
    ):
        """Native Gen 5 multi-send forward pass using the layer-6 refined edge track and WDL value head."""
        h, e = self.encode_planets_and_edges(planet_feats, timeline_feats, edge_feats, pad_mask)
        ms_logits, ms_fracs = self.predict_multi_send(h, edge_feats, edge_hidden=e)
        wdl_logits = self.decode_wdl_from_hidden(h, pad_mask)
        wdl_p = F.softmax(wdl_logits, dim=-1)
        value = (wdl_p[:, 2] - wdl_p[:, 0] - self.contempt * wdl_p[:, 1]).clamp(-1.0, 1.0)
        if return_wdl:
            if return_aux and return_hidden:
                return ms_logits, ms_fracs, value, wdl_logits, self.predict_aux_future(h), h
            if return_aux:
                return ms_logits, ms_fracs, value, wdl_logits, self.predict_aux_future(h)
            return ms_logits, ms_fracs, value, wdl_logits
        if return_aux and return_hidden:
            return ms_logits, ms_fracs, value, self.predict_aux_future(h), h
        if return_aux:
            return ms_logits, ms_fracs, value, self.predict_aux_future(h)
        if return_hidden:
            return ms_logits, ms_fracs, value, h
        return ms_logits, ms_fracs, value

    def forward(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        return_aux: bool = False,
        return_hidden: bool = False,
    ):
        """Compatibility forward pass deriving single-target logits from native Gen 5 multi-send logits."""
        h, e = self.encode_planets_and_edges(planet_feats, timeline_feats, edge_feats, pad_mask)
        ms_logits, ms_fracs = self.predict_multi_send(h, edge_feats, edge_hidden=e)
        value = self.decode_value_from_hidden(h, pad_mask)
        # Synthesize conservative hold logit = 0.0 vs ms_logits so single-target fallback stays coherent
        hold_logit = torch.zeros_like(ms_logits[:, :, :1])
        action_logits = torch.cat([hold_logit, ms_logits], dim=-1)
        if return_aux and return_hidden:
            return action_logits, ms_fracs, value, self.predict_aux_future(h), h
        if return_aux:
            return action_logits, ms_fracs, value, self.predict_aux_future(h)
        if return_hidden:
            return action_logits, ms_fracs, value, h
        return action_logits, ms_fracs, value


def build_galcon_gen5_transformer(
    *,
    d_model: int = 192,
    n_heads: int = 6,
    n_layers: int = 6,
    dim_ff: int = 384,
    edge_hidden: int = 24,
    num_val_queries: int = 4,
    contempt: float = 0.0,
    dropout: float = 0.08,
) -> GalconGen5Transformer:
    """Instantiate the Gen 5 (~2.93M) Edge-Gated Galcon Transformer with WDL value head."""
    return GalconGen5Transformer(
        planet_dim=22,
        timeline_dim=8,
        edge_dim=12,
        redir_dim=10,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        edge_hidden=edge_hidden,
        num_val_queries=num_val_queries,
        feature_version=2,
        contempt=contempt,
        dropout=dropout,
    )


class _FastGen5Block(nn.Module):
    """Fused inference-only variant of ``EdgeGatedTransformerBlock`` for unpadded CPU search batches.

    Fuses paired linear projections ``(edge_bias, edge_val)``, ``(out_proj, edge_msg_out)``, and
    ``(edge_src, edge_dst)``, omits no-op eval ``Dropout`` calls, and exposes ``forward_node_only``
    so value-only leaf evaluation skips the unused final-layer edge refinement.
    """

    def __init__(self, blk: EdgeGatedTransformerBlock) -> None:
        super().__init__()
        D = int(blk.d_model)
        H = int(blk.n_heads)
        Dh = int(blk.head_dim)
        De = int(blk.edge_hidden)
        self.d_model: int = D
        self.n_heads: int = H
        self.head_dim: int = Dh
        self.edge_hidden: int = De
        self.scale: float = float(blk.scale)

        self.norm1 = copy.deepcopy(blk.norm1)
        self.qkv = copy.deepcopy(blk.qkv)

        self.edge_bv = nn.Linear(De, H + De, bias=True)
        self.edge_bv.weight.copy_(
            torch.cat([blk.edge_bias.weight.detach(), blk.edge_val.weight.detach()], dim=0)
        )
        self.edge_bv.bias.copy_(
            torch.cat([blk.edge_bias.bias.detach(), blk.edge_val.bias.detach()], dim=0)
        )

        self.attn_out = nn.Linear(D + H * De, D, bias=True)
        self.attn_out.weight.copy_(
            torch.cat([blk.out_proj.weight.detach(), blk.edge_msg_out.weight.detach()], dim=1)
        )
        self.attn_out.bias.copy_(blk.out_proj.bias.detach())

        self.norm2 = copy.deepcopy(blk.norm2)
        self.w_gate_up = copy.deepcopy(blk.ffn.w_gate_up)
        self.w_down = copy.deepcopy(blk.ffn.w_down)

        self.edge_norm = copy.deepcopy(blk.edge_norm)
        self.edge_sd = nn.Linear(D, 2 * De, bias=False)
        self.edge_sd.weight.copy_(
            torch.cat([blk.edge_src.weight.detach(), blk.edge_dst.weight.detach()], dim=0)
        )
        self.edge_upd_lin = copy.deepcopy(blk.edge_upd[1])

    def forward_node_only(self, x: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        N = x.shape[1]
        D = self.d_model
        H = self.n_heads
        Dh = self.head_dim
        De = self.edge_hidden

        xn = self.norm1(x)
        qkv = self.qkv(xn).view(B, N, 3, H, Dh)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        ebv = self.edge_bv(e)
        eb = ebv[..., :H].permute(0, 3, 1, 2)
        ev = ebv[..., H:]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale + eb
        attn = F.softmax(logits, dim=-1)

        node_ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        attn_bn = attn.transpose(1, 2).reshape(B * N, H, N)
        ev_bn = ev.reshape(B * N, N, De)
        edge_ctx = torch.bmm(attn_bn, ev_bn).reshape(B, N, H * De)
        x = x + self.attn_out(torch.cat([node_ctx, edge_ctx], dim=-1))

        xn2 = self.norm2(x)
        gu = self.w_gate_up(xn2)
        half = gu.shape[-1] // 2
        x = x + self.w_down(F.silu(gu[..., :half]) * gu[..., half:])
        return x, xn2

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, xn2 = self.forward_node_only(x, e)
        De = self.edge_hidden
        esd = self.edge_sd(xn2)
        e_in = self.edge_norm(e) + esd[..., :De].unsqueeze(2) + esd[..., De:].unsqueeze(1)
        e = e + self.edge_upd_lin(F.silu(e_in))
        return x, e


class FastGen5Infer(nn.Module):
    """Dynamic-shape TorchScript-optimized CPU inference module for ``GalconGen5Transformer``.

    Preserves exact float32 numerical parity (``<1e-5``) across arbitrary ``(B, N, T)`` shapes
    while eliminating Python dispatch overhead, fusing paired linear projections, and skipping
    dead edge-refinement work during value-only leaf evaluation.
    """

    def __init__(self, model: GalconGen5Transformer) -> None:
        super().__init__()
        D = int(model.d_model)
        De = int(model.edge_hidden)
        ms_hid = D // 2
        edge_dim = int(model.ms_raw_edge_proj.in_features)
        self.d_model: int = D
        self.ms_hid: int = ms_hid

        self.timeline_conv = copy.deepcopy(model.timeline_conv)
        self.node_in = copy.deepcopy(model.node_in)
        self.edge_in = copy.deepcopy(model.edge_in)

        self.layers = nn.ModuleList([_FastGen5Block(blk) for blk in model.layers[:-1]])
        self.last_layer = _FastGen5Block(model.layers[-1])
        self.final_norm = copy.deepcopy(model.final_norm)
        self.final_edge_norm = copy.deepcopy(model.final_edge_norm)

        self.ms_sd = nn.Linear(D, 2 * ms_hid, bias=True)
        self.ms_sd.weight.copy_(
            torch.cat([model.ms_src_proj.weight.detach(), model.ms_dst_proj.weight.detach()], dim=0)
        )
        self.ms_sd.bias.copy_(
            torch.cat([model.ms_src_proj.bias.detach(), model.ms_dst_proj.bias.detach()], dim=0)
        )

        self.ms_edge_both = nn.Linear(De + edge_dim, ms_hid, bias=True)
        self.ms_edge_both.weight.copy_(
            torch.cat([model.ms_edge_proj.weight.detach(), model.ms_raw_edge_proj.weight.detach()], dim=1)
        )
        self.ms_edge_both.bias.copy_(model.ms_edge_proj.bias.detach() + model.ms_raw_edge_proj.bias.detach())
        self.ms_head = copy.deepcopy(model.ms_head)

        self.register_buffer(
            "val_queries_t",
            (model.val_queries.detach().t() * (D ** -0.5)).contiguous(),
        )
        self.val_kv = nn.Linear(D, 2 * D, bias=False)
        self.val_kv.weight.copy_(
            torch.cat([model.val_key.weight.detach(), model.val_val.weight.detach()], dim=0)
        )
        self.val_wdl_head = copy.deepcopy(model.val_wdl_head)

    def _embed(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = timeline_feats.shape[0]
        N = timeline_feats.shape[1]
        T = timeline_feats.shape[2]
        Ct = timeline_feats.shape[3]
        tl = timeline_feats.reshape(B * N, T, Ct).transpose(1, 2)
        tl_h = self.timeline_conv(tl)
        tl_pool = torch.cat([tl_h.mean(dim=-1), tl_h.amax(dim=-1), tl_h[:, :, -1]], dim=-1).view(B, N, 192)
        x = self.node_in(torch.cat([planet_feats, tl_pool], dim=-1))
        e = self.edge_in(edge_feats)
        return x, e

    def _decode_value(self, h: torch.Tensor, contempt: float) -> torch.Tensor:
        B = h.shape[0]
        D = self.d_model
        kv = self.val_kv(h)
        k = kv[..., :D]
        v = kv[..., D:]
        attn_logits = torch.matmul(k, self.val_queries_t).transpose(1, 2)
        attn_w = F.softmax(attn_logits, dim=-1)
        q_pool = torch.matmul(attn_w, v).reshape(B, -1)
        mean_pool = h.mean(dim=1)
        wdl = F.softmax(self.val_wdl_head(torch.cat([q_pool, mean_pool], dim=-1)), dim=-1)
        return (wdl[:, 2] - wdl[:, 0] - contempt * wdl[:, 1]).clamp(-1.0, 1.0)

    def _decode_ms(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_ms = self.ms_hid
        sd = self.ms_sd(h)
        pair = (
            sd[..., :H_ms].unsqueeze(2)
            + sd[..., H_ms:].unsqueeze(1)
            + self.ms_edge_both(torch.cat([e, edge_feats], dim=-1))
        )
        out = self.ms_head(pair)
        return out[..., 0], torch.sigmoid(out[..., 1])

    def forward(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        contempt: float,
    ) -> torch.Tensor:
        return self.eval_values(planet_feats, timeline_feats, edge_feats, contempt)

    @torch.jit.export
    def eval_values(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        contempt: float,
    ) -> torch.Tensor:
        x, e = self._embed(planet_feats, timeline_feats, edge_feats)
        for layer in self.layers:
            x, e = layer(x, e)
        x, _ = self.last_layer.forward_node_only(x, e)
        h = self.final_norm(x)
        return self._decode_value(h, contempt)

    @torch.jit.export
    def eval_ms(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        contempt: float,
        need_value: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, e = self._embed(planet_feats, timeline_feats, edge_feats)
        for layer in self.layers:
            x, e = layer(x, e)
        x, e = self.last_layer(x, e)
        h = self.final_norm(x)
        e = self.final_edge_norm(e)
        ms_logits, ms_fracs = self._decode_ms(h, e, edge_feats)
        if need_value:
            vals = self._decode_value(h, contempt)
        else:
            vals = torch.zeros((planet_feats.shape[0],), dtype=planet_feats.dtype, device=planet_feats.device)
        return ms_logits, ms_fracs, vals

    @torch.jit.export
    def eval_ms_with_hidden(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        contempt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, e = self._embed(planet_feats, timeline_feats, edge_feats)
        for layer in self.layers:
            x, e = layer(x, e)
        x, e = self.last_layer(x, e)
        h = self.final_norm(x)
        e = self.final_edge_norm(e)
        ms_logits, ms_fracs = self._decode_ms(h, e, edge_feats)
        vals = self._decode_value(h, contempt)
        return ms_logits, ms_fracs, vals, h


def build_fast_gen5_infer(model: GalconGen5Transformer) -> Optional[torch.jit.ScriptModule]:
    """Compile a fused TorchScript CPU inference wrapper for ``GalconGen5Transformer``."""
    if not isinstance(model, GalconGen5Transformer) or len(model.layers) < 1:
        return None
    try:
        with torch.no_grad():
            fast = FastGen5Infer(model).eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scripted = torch.jit.script(fast)
            optimized = torch.jit.optimize_for_inference(
                scripted,
                other_methods=["eval_values", "eval_ms", "eval_ms_with_hidden"],
            )
        return optimized
    except Exception:
        return None

