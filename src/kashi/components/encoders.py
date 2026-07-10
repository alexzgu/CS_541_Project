"""Frame-level feature encoders. Swap via [pipeline].encoder.

The wav2vec2 encoder accepts ANY HuggingFace wav2vec2/HuBERT-family checkpoint
(encoder.wav2vec2.checkpoint), e.g. facebook/wav2vec2-base,
rinna/japanese-wav2vec2-base, rinna/japanese-hubert-base. If a trained
contrastive projection head is configured, its output space is what the rest
of the pipeline sees.
"""

from __future__ import annotations

import numpy as np

from ..registry import register


@register("encoder", "wav2vec2")
class Wav2Vec2Encoder:
    frame_ms = 20

    def __init__(self, cfg):
        self.checkpoint = cfg["encoder.wav2vec2.checkpoint"]
        self.chunk_s = float(cfg.get("encoder.wav2vec2.chunk_s", 20.0))
        self.projection_head = cfg.get("encoder.wav2vec2.projection_head", "")
        self._device_pref = cfg.get("encoder.wav2vec2.device", "auto")
        self._model = None
        self._proj = None
        self._dim = 768

    @property
    def dim(self) -> int:
        if self._proj is not None:
            return self._proj.out_dim
        return self._dim

    def _load(self):
        if self._model is not None:
            return
        import torch

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise SystemExit(
                "encoder 'wav2vec2' needs the [hf] extra: pip install 'kashi[hf]'"
            ) from e
        device = (
            "cuda" if self._device_pref == "auto" and torch.cuda.is_available() else
            ("cpu" if self._device_pref == "auto" else self._device_pref)
        )
        try:  # memory-efficient attention lets full songs fit on 6 GB GPUs
            self._model = AutoModel.from_pretrained(
                self.checkpoint, attn_implementation="sdpa"
            ).to(device).eval()
        except (TypeError, ValueError):
            self._model = AutoModel.from_pretrained(self.checkpoint).to(device).eval()
        if device != "cpu":
            self._model = self._model.half()  # full songs in 6 GB; features stored fp32
        self._dim = int(self._model.config.hidden_size)
        self._device = device
        if self.projection_head:
            from ..nn.contrastive import ProjectionHead

            self._proj = ProjectionHead.load(self.projection_head).to(device).eval()

    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """mono float32 waveform -> [T, dim] features on the 20 ms grid.

        Matches the legacy encoding semantics (models/wave2vec2/wave.py): the
        waveform is zero-mean/unit-variance normalised over the WHOLE song
        (what Wav2Vec2Processor does) and, when memory allows, run through the
        model in one forward pass (global attention context). On CPU or very
        long inputs it falls back to normalised overlapping chunks.
        """
        import torch

        self._load()
        hop = int(sr * self.frame_ms / 1000)
        target_T = len(wave) // hop
        if target_T == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        x = torch.from_numpy(wave).float()
        x = (x - x.mean()) / (x.std() + 1e-7)
        dtype = next(self._model.parameters()).dtype

        full_ok = self._device != "cpu" or len(wave) <= int(60 * sr)
        with torch.inference_mode():
            h = None
            if full_ok:
                try:
                    h = self._model(x[None].to(self._device, dtype)).last_hidden_state[0]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if h is None:
                # overlapping chunks; centre parts stitched
                chunk = int(self.chunk_s * sr)
                ov = int(1.0 * sr)
                parts: list[torch.Tensor] = []
                s = 0
                while s < len(x):
                    piece = x[max(0, s - ov) : s + chunk + ov]
                    if len(piece) < hop:
                        break
                    hh = self._model(piece[None].to(self._device, dtype)).last_hidden_state[0]
                    lead = (s - max(0, s - ov)) // hop
                    keep = min(chunk // hop, len(hh) - lead)
                    parts.append(hh[lead : lead + keep])
                    s += chunk
                h = torch.cat(parts, dim=0)
            h = h.float()
            if self._proj is not None:
                h = self._proj(h)
        feats = h.cpu().numpy()
        if len(feats) > target_T:
            feats = feats[:target_T]
        elif len(feats) < target_T:
            feats = np.pad(feats, ((0, target_T - len(feats)), (0, 0)), mode="edge")
        return feats.astype(np.float32)


@register("encoder", "mel")
class MelEncoder:
    """Log-mel spectrogram on the same frame grid (no downloads; CPU-cheap)."""

    def __init__(self, cfg):
        self.n_mels = int(cfg["encoder.mel.n_mels"])
        self.frame_ms = cfg.frame_ms
        self._transform = None
        self._sr = None

    @property
    def dim(self) -> int:
        return self.n_mels

    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray:
        import torch
        import torchaudio

        hop = int(sr * self.frame_ms / 1000)
        if self._transform is None or self._sr != sr:
            self._transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_mels=self.n_mels, hop_length=hop
            )
            self._sr = sr
        with torch.inference_mode():
            spec = self._transform(torch.from_numpy(wave).float())
        spec = torch.log(spec + 1e-9).T  # [T, n_mels]
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
        T = len(wave) // hop
        return spec[:T].numpy().astype(np.float32)
