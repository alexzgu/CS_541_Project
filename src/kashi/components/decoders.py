"""Semi-Markov segmental decoder (spec §6) — textless joint inference.

Score of a labeled segmentation:
  sum_k [ lambda_c * sum_t log p_f(t_k | x_t)   (Model 2f, cumulative sums)
        + lambda_d * log P_dur(d_k | t_k)       (NB duration prior)
        + lambda_b * beta_{s_k}                 (Model-1 boundary logit, optional)
        + lambda_lm * log A(t_k | t_{k-1}) ]    (token bigram, optional)

Exact Viterbi over all segmentations; long low-energy runs split the song into
independently-decoded chunks (the silence-duration exemption). No transcript
is consumed anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..registry import register
from ..subtitles import Segment
from ..tokens import SILENCE_ID, TOKENS
from .base import FrameAux


def _latest_frame_checkpoint(cfg) -> Path | None:
    runs = sorted(cfg.runs_dir.glob("frame-*/best.pt"))
    return runs[-1] if runs else None


@register("decoder", "segmental")
class SegmentalDecoder:
    def __init__(self, cfg):
        import torch

        from ..nn.classifier import FrameClassifier
        from ..stats.durations import log_duration_table
        from ..stats.lm import log_bigram

        self.cfg = cfg
        sect = cfg.section("decoder.segmental")
        self.d_max = int(sect.get("d_max", 60))
        self.lam_c = float(sect.get("lambda_c", 1.0))
        self.lam_d = float(sect.get("lambda_d", 0.5))
        self.lam_b = float(sect.get("lambda_b", 0.5))
        self.lam_lm = float(sect.get("lambda_lm", 0.0))

        self.emissions = sect.get("emissions", "frame")   # frame | ctc
        self.frame = None
        self._ctc = None
        if self.emissions == "ctc":
            self.ctc_dir = sect.get("ctc_model", "artifacts/ctc_local/out/ctc_model")
        else:
            ckpt = sect.get("frame_checkpoint", "") or _latest_frame_checkpoint(cfg)
            if not ckpt or not Path(ckpt).is_file():
                raise SystemExit(
                    "decoder 'segmental' needs a trained frame classifier: run "
                    "`kashi train frame`, then set decoder.segmental.frame_checkpoint"
                )
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.frame = FrameClassifier(**payload["hparams"])
            self.frame.load_state_dict(payload["state_dict"])
            self.frame.eval()

        self.log_dur = log_duration_table(cfg, self.d_max)      # [C, d_max]
        d_min = int(sect.get("d_min", 2))
        if d_min > 1:  # morae are >= ~50 ms; silence may stay short
            self.log_dur[:, : d_min - 1] = -1e9
            self.log_dur[SILENCE_ID, : d_min - 1] = np.log(1e-4)
        self.log_A = log_bigram(cfg) if self.lam_lm > 0 else None
        if self.lam_lm > 0 and self.log_A is None:
            raise SystemExit("lambda_lm > 0 but no bigram — run `kashi fit lm`")

    # ------------------------------------------------------------------
    def _viterbi_chunk(self, logp: np.ndarray, beta: np.ndarray) -> list[tuple[int, int, int]]:
        """Exact DP over one chunk. logp: [n, C] frame log-posteriors.
        Returns [(start, end, class)] covering [0, n)."""
        n, C = logp.shape
        Dm = min(self.d_max, n)
        cum = np.vstack([np.zeros((1, C)), np.cumsum(logp, axis=0)])  # [n+1, C]
        dur = self.lam_d * self.log_dur[:, :Dm].T                     # [Dm, C]

        M = np.full(n + 1, -np.inf)
        M[0] = 0.0
        # W[t] = per-class best predecessor value at boundary t
        W = np.full((n + 1, C), -np.inf) if self.lam_lm > 0 else None
        if W is not None:
            W[0] = 0.0
        back_d = np.zeros(n + 1, dtype=int)
        back_u = np.zeros(n + 1, dtype=int)
        prev_u = np.zeros((n + 1, C), dtype=int) if W is not None else None

        for e in range(1, n + 1):
            dmax_e = min(Dm, e)
            starts = e - np.arange(1, dmax_e + 1)                     # [d]
            span = self.lam_c * (cum[e][None, :] - cum[starts])       # [d, C]
            total = span + dur[:dmax_e] + self.lam_b * beta[starts][:, None]
            if W is None:
                total = total + M[starts][:, None]
                flat = int(np.argmax(total))
                d_star, u_star = flat // C, flat % C
                M[e] = total[d_star, u_star]
                back_d[e], back_u[e] = d_star + 1, u_star
            else:
                # V[e, u] = max_d ( total[d, u] + W[start_d, u] )
                cand = total + W[starts]                              # W already folds bigram
                d_star = np.argmax(cand, axis=0)                      # [C]
                V_e = cand[d_star, np.arange(C)]
                M[e] = V_e.max()
                back_u[e] = int(V_e.argmax())
                back_d[e] = d_star[back_u[e]] + 1
                # fold transitions for successors: W[e, u] = max_u' V[e,u'] + lam*A[u',u]
                scores = V_e[:, None] + self.lam_lm * self.log_A     # [u', u]
                prev_u[e] = np.argmax(scores, axis=0)
                W[e] = scores[prev_u[e], np.arange(C)]

        # backtrack
        segs: list[tuple[int, int, int]] = []
        e = n
        u = int(back_u[e])
        while e > 0:
            d = int(back_d[e])
            segs.append((e - d, e, u))
            if prev_u is not None:
                u_next = int(prev_u[e - d, u]) if e - d > 0 else u
            e -= d
            if W is None:
                u = int(back_u[e])
            else:
                u = u_next
        segs.reverse()
        return segs

    # ------------------------------------------------------------------
    def _ctc_log_probs(self, wave: np.ndarray, sr: int, T: int) -> np.ndarray:
        """Emissions straight from the CTC model (blank = 109 = <silence>)."""
        import torch

        if self._ctc is None:
            from transformers import AutoModelForCTC

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._ctc = AutoModelForCTC.from_pretrained(self.ctc_dir).to(dev).eval()
            self._ctc_dev = dev
        w = (wave - wave.mean()) / (wave.std() + 1e-7)
        hop = sr * self.cfg.frame_ms // 1000
        chunk, ov = int(30.0 * sr), int(2.0 * sr)
        parts, step = [], chunk - ov
        with torch.inference_mode():
            for s in range(0, len(w), step):
                piece = torch.from_numpy(w[max(0, s - ov): s + chunk].copy()).float()
                if self._ctc_dev == "cuda":
                    with torch.autocast("cuda", torch.float16):
                        lg = self._ctc(piece[None].cuda()).logits[0].float().cpu()
                else:
                    lg = self._ctc(piece[None]).logits[0]
                lead = (s - max(0, s - ov)) // hop
                keep = min((chunk - ov) // hop, len(lg) - lead)
                parts.append(lg[lead: lead + keep])
                if s + chunk >= len(w):
                    break
        em = torch.log_softmax(torch.cat(parts), -1).numpy()
        if len(em) < T:
            em = np.pad(em, ((0, T - len(em)), (0, 0)), mode="edge")
        return em[:T]

    def _spike_decode(self, logp: np.ndarray, frame_s: float,
                      max_dur_s: float = 0.6, onset_shift: int = -1) -> list[Segment]:
        """Greedy CTC decoding of peaky emissions (textless): collapse repeats,
        drop blank; token onset = first frame of its run (the spike), extent =
        up to the next onset capped at max_dur_s. Confidence = spike posterior."""
        path = logp.argmax(-1)
        probs = np.exp(logp[np.arange(len(path)), path])
        spikes: list[tuple[int, int, float]] = []   # (frame, class, prob)
        prev = -1
        for t, c in enumerate(path):
            if c != prev and c != SILENCE_ID:
                spikes.append((t, int(c), float(probs[t])))
            prev = c
        out: list[Segment] = []
        for k, (t, c, p) in enumerate(spikes):
            t0 = max(0, t + onset_shift)   # CTC spikes fire ~1 frame late
            end_f = spikes[k + 1][0] + onset_shift if k + 1 < len(spikes) else t0 + int(max_dur_s / frame_s)
            end_f = min(max(t0 + 1, end_f), t0 + int(max_dur_s / frame_s))
            out.append(Segment(t0 * frame_s, end_f * frame_s,
                               TOKENS[c], confidence=round(p, 3)))
        return out

    def decode(self, feats: np.ndarray, aux: FrameAux | None = None) -> list[Segment]:
        frame_s = self.cfg.frame_ms / 1000.0
        T = len(feats)
        if self.emissions == "ctc":
            if aux is None or "wave" not in aux.extras:
                raise SystemExit("decoder emissions=ctc needs the waveform (pipeline provides it)")
            logp = self._ctc_log_probs(aux.extras["wave"], aux.extras.get("sr", 16000), T)
            return self._spike_decode(logp, frame_s)
        logp = self.frame.log_probs(feats)                            # [T, C]
        beta = np.zeros(T)
        if aux is not None and aux.boundary_logits is not None:
            beta = np.asarray(aux.boundary_logits)[:T]

        # chunk at long low-energy runs (> 3 s): decode voiced islands
        if aux is not None and aux.rms_db is not None and len(aux.rms_db):
            voiced = aux.rms_db[:T] > (aux.rms_db[:T].max() - 45.0)
        else:
            voiced = np.ones(T, dtype=bool)
        min_gap = int(3.0 / frame_s)
        chunks: list[tuple[int, int]] = []
        t = 0
        while t < T:
            if not voiced[t]:
                u = t
                while u < T and not voiced[u]:
                    u += 1
                if u - t >= min_gap:
                    t = u
                    continue
                t = u
            else:
                s = t
                while t < T:
                    if voiced[t]:
                        t += 1
                    else:
                        u = t
                        while u < T and not voiced[u]:
                            u += 1
                        if u - t >= min_gap:
                            break
                        t = u
                chunks.append((s, t))
        if not chunks:
            chunks = [(0, T)]

        out: list[Segment] = []
        for a, b in chunks:
            for s, e, u in self._viterbi_chunk(logp[a:b], beta[a:b]):
                if u == SILENCE_ID:
                    continue  # silences re-derived by the pipeline gap-fill
                conf = float(np.exp(logp[a + s : a + e, u].mean()))
                out.append(Segment((a + s) * frame_s, (a + e) * frame_s,
                                   TOKENS[u], confidence=round(conf, 3)))
        return out
