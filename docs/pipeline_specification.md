# kashi v2: Mathematical Specification of the Textless Syllable-Level Transcription Pipeline

Companion to *Textless Syllable Alignment of Japanese Songs* (CS 541 report) and to `ROADMAP.md`. 2026-07-09.

**Scope.** Complete mathematical definition of every model in the pipeline: vocal separation, frame features (wav2vec2, contrastive projection, voicing/TDA), Model 1 (boundary scorer, corrected), Model 2 (syllable classifier, segment and frame variants), the phonetic kernel, the semi-Markov segmental decoder that joins them, the sticky HDP-HMM/HSMM used for unsupervised structure, the alignment-free dataset-cleaning procedure, unsupervised objectives, and evaluation metrics. Design constraints throughout: (i) *textless* — no transcript is available or used at inference; (ii) *no forced alignment anywhere*, including dataset cleaning (Remark 1); (iii) deep networks supply local evidence, while segmentation, duration, and label structure are explicit probabilistic components rather than black-box behavior.

## 1. Notation and problem statement

Notation follows the report; collisions in the report are resolved as noted.

| Symbol | Meaning | Report correspondence |
|---|---|---|
| $a \in \mathbb{R}^{N_a}$ | mono waveform, $f_s = 16{,}000$ Hz | (audio file) |
| $h = 0.02$ s | frame hop; $T = \lfloor N_a/(f_s h)\rfloor$ frames | 20-ms interval |
| $x_t \in \mathbb{R}^{768}$ | wav2vec2 vector of frame $t \in \{1,\dots,T\}$ | $x_t$ |
| $\tilde{x}_t \in \mathbb{R}^{d}$ | final frame feature (§3) | — (new) |
| $\mathcal{T}$, $\lvert\mathcal{T}\rvert = 110$ | token set: 109 hiragana + $\langle\mathrm{sil}\rangle$ | $T$ (set) [^1] |
| $y_t \in [0,1]$ | Model-1 per-frame break score | $y_t$ |
| $\sigma_k = (t_k, s_k, e_k)$ | segment $k$: token $t_k \in \mathcal{T}$, start $s_k$, end $e_k$ (s) | $s_k = (t_k, s_k, e_k)$ [^2] |
| $\sigma = (\sigma_1,\dots,\sigma_K)$ | full segmentation; $s_1 = 0$, $s_{k+1} = e_k$, $e_K = Th$ | $\{s_k\}_{k=0}^{K}$ |

[^1]: The report uses $T$ for both the token set and the sequence length in $f(x_1,\dots,x_T)$; we write $\mathcal{T}$ for the set and keep $T$ for length.
[^2]: The report overloads $s_k$ as both the segment and its start time; we write $\sigma_k$ for the segment.

**Task.** Learn $F: a \mapsto \sigma$ from a corpus $\mathcal{D} = \{(a^{(i)}, \hat{\sigma}^{(i)})\}_{i=1}^{n}$ whose label **times** are corrupted,

$$\hat{s}_k = s_k^\star + \varepsilon_k, \qquad |\varepsilon_k| \le \varepsilon_{\max} \approx 50\text{ ms} \;(= 2.5\text{ frames}), \tag{1}$$

whose **sequences** $(t_1,\dots,t_K)$ are reliable, and whose $\langle\mathrm{sil}\rangle$ segments may contain unlabeled vocal events (breaths). At inference only $a$ is given.

> **Remark 1 (exclusion of forced alignment).** Forced alignment solves, for a *given* token sequence $w_{1:M}$,
> $$\hat{A} = \arg\max_{A \in \mathcal{A}(w_{1:M}, T)} p(x_{1:T}, A \mid w_{1:M})$$
> over monotone alignments $\mathcal{A}$. Every procedure below is required **not** to solve this problem or any restriction of it: no stage conditions a time-inference on a token sequence. Training may use the labeled sequences as supervision targets (all supervised learning does); what is excluded is deriving *timestamps* by aligning a transcript — at inference (no transcript exists; singers go off script) and during dataset cleaning (§8 uses only sequence-free local corrections).

**Pipeline operator.** $F = W \circ D \circ \Phi \circ V$: separation $V$ (§2), frame features $\Phi$ (§3), segmental decode $D$ (§6) consuming the model scores of §§4–5, subtitle writing $W$ (exact; not modeled). Each stage is a swappable component; this document specifies the default mathematics of each slot.

## 2. Stage V: vocal separation (off the shelf)

With $X = \mathrm{STFT}(a) \in \mathbb{C}^{F \times N}$, a separator computes a vocal estimate $\hat{a} = \mathrm{iSTFT}(M_\theta(X) \odot X)$ where $M_\theta \in [0,1]^{F\times N}$ (or complex-valued) is produced by a pretrained network (default: Mel-Band RoFormer checkpoint; alternates: Demucs-family time-domain models, identity). $V$ is a fixed operator: its parameters are never trained here, and the report verified label preservation on the corpus. All downstream stages see only $\hat{a}$, resampled to 16 kHz mono.

## 3. Stage Φ: frame representation

### 3.1 wav2vec2 backbone

A pretrained wav2vec2/HuBERT encoder maps $\hat a$ to latents $z_\tau$ (convolutional encoder, 20-ms stride, ≈25-ms receptive field) and context vectors

$$x_t = \mathrm{TransformerEncoder}(z)_t \in \mathbb{R}^{768}, \qquad t = 1,\dots,T .$$

The checkpoint is a config choice (default candidates: `rinna/japanese-hubert-base`, `rinna/japanese-wav2vec2-base`; legacy: `facebook/wav2vec2-base`). The backbone is frozen in all default configurations.

### 3.2 Learned projection (contrastive; optional)

$g_w: \mathbb{R}^{768} \to \mathbb{R}^{128}$,

$$g_w(x) = \frac{W_2\,\phi(W_1 x + b_1) + b_2}{\lVert W_2\,\phi(W_1 x + b_1) + b_2 \rVert_2},
\qquad W_1 \in \mathbb{R}^{256\times768},\; W_2 \in \mathbb{R}^{128\times256},\; \phi = \mathrm{GELU},$$

trained by the objectives of §9 on cached features (backbone frozen). When enabled, $g_w(x_t)$ replaces $x_t$ everywhere downstream (deep-kernel view: $g_w$ *is* the learned similarity).

### 3.3 Voicing score $v_t$ (periodicity)

Let $w_t \in \mathbb{R}^{L_w}$ be the waveform window of 46 ms centered on frame $t$, mean-removed. Normalized autocorrelation and voicing:

$$\rho_t(\ell) = \frac{\sum_{n} w_t[n]\, w_t[n+\ell]}{\sqrt{\sum_n w_t[n]^2 \sum_n w_t[n+\ell]^2}},
\qquad
v_t = \max_{\ell \in [f_s/f_{\max},\, f_s/f_{\min}]} \rho_t(\ell),
\quad (f_{\min}, f_{\max}) = (70, 500)\,\mathrm{Hz}. \tag{2}$$

Voiced morae (vowels; voiced consonants *ga, wo*, …) are quasi-periodic ⇒ $v_t \to 1$; breath/silence are aperiodic ⇒ $v_t$ small. **TDA variant** (interchangeable estimator of the same quantity): Takens delay embedding $y_n = (w_t[n], w_t[n+\tau], w_t[n+2\tau]) \in \mathbb{R}^3$ with $\tau = f_s/(2 f_{\max})$; Vietoris–Rips filtration on $\{y_n\}$; with $\mathrm{PD}_1$ the $H_1$ persistence diagram,

$$v_t^{\mathrm{TDA}} = \frac{\max_{(b,d) \in \mathrm{PD}_1} (d-b)}{\mathrm{diam}\{y_n\}} \in [0,1],$$

i.e. the longest normalized loop lifetime (periodic ⇔ closed orbit ⇔ persistent $H_1$ class). Default is (2) (cheap, no dependency); the TDA estimator is a config alternate benchmarked once.

### 3.4 Final frame feature

$$\tilde{x}_t = \big[\, g_w(x_t) \text{ (or } x_t\text{)};\; v_t;\; e_t \,\big], \qquad e_t = \log \mathrm{RMS}(w_t),$$

so $d \in \{770, 130\}$ depending on projection use. The scalar channels give every downstream model explicit, interpretable voicing/energy evidence (targets the *ka/ga*, *wo/o* confusion axis and breath detection).

## 4. Model 1: boundary scorer

### 4.1 Architecture (corrected)

Per-frame boundary logits from a temporal transformer:

$$u_t^{(0)} = W_{\mathrm{in}} \tilde{x}_t + p_t, \qquad W_{\mathrm{in}} \in \mathbb{R}^{d_m \times d},\; d_m = 256,$$

$$p_t[2i] = \sin\big(t / 10000^{2i/d_m}\big), \qquad p_t[2i+1] = \cos\big(t / 10000^{2i/d_m}\big),$$

for layers $\ell = 1,\dots,L$ ($L=2$), with $U = [u_1;\dots;u_T]$ processed **as a length-$T$ sequence**:

$$\mathrm{head}_i(U) = \mathrm{softmax}\Big(\tfrac{(U W_i^Q)(U W_i^K)^\top}{\sqrt{d_k}}\Big)\, U W_i^V, \qquad i = 1,\dots,H,\; H = 8,\; d_k = d_m/H,$$

$$\mathrm{MHA}(U) = [\mathrm{head}_1;\dots;\mathrm{head}_H]\, W^{O}, \qquad
U' = \mathrm{LN}\big(U + \mathrm{MHA}(U)\big), \qquad
U^{(\ell)} = \mathrm{LN}\big(U' + \mathrm{FFN}(U')\big),$$

$$\mathrm{FFN}(u) = W_2^{\mathrm{ff}}\,\mathrm{ReLU}(W_1^{\mathrm{ff}} u + b_1) + b_2, \qquad W_1^{\mathrm{ff}} \in \mathbb{R}^{512 \times d_m},$$

$$\beta_t = w_{\mathrm{out}}^\top u_t^{(L)}, \qquad y_t = \sigma(\beta_t) = \big(1+e^{-\beta_t}\big)^{-1}.$$

Attention masking limits context to $\pm W_a = \pm 100$ frames (±2 s; boundaries are local evidence). **Corrections vs. the recovered legacy implementation** (`google drive/Code/Transformer_Wave2Vec_AAAAA.ipynb`): (i) attention must run over the *time* axis (the legacy code omitted `batch_first=True`, so self-attention saw length-1 sequences — no temporal context); (ii) positional encodings $p_t$ were absent; (iii) label expansion must be edge-guarded (below).

### 4.2 Label construction and losses

Break frames from labels: $\hat{b}_j = \mathrm{frame}(\hat{e}_{k_j})$ for internal segment ends, $j = 1,\dots,J$, excluding excluded rows. Two supervised objectives (both implemented; selected by the metric in §10):

**(a) Soft-kernel BCE (report's method, regularized).** Targets by edge-guarded kernel placement,

$$\tilde{y}_\tau = \min\Big(1, \sum_{j=1}^J \kappa(\tau - \hat{b}_j)\Big), \qquad
\kappa(-2{:}3) = (0.2,\, 0.35,\, 1,\, 1,\, 0.8,\, 0.25),\; \kappa = 0 \text{ otherwise},$$

$$\mathcal{L}_{\mathrm{soft}} = -\tfrac{1}{T}\textstyle\sum_{\tau=1}^{T} m_\tau \big[ w_+ \tilde{y}_\tau \log y_\tau + (1-\tilde{y}_\tau)\log(1-y_\tau) \big],$$

$m_\tau \in \{0,1\}$ masking excluded/padded frames, $w_+ =$ (#neg/#pos) on unexpanded labels.

**(b) Latent-offset marginalization (principled jitter model; new).** Model (1) directly: the true break is $b_j = \hat{b}_j + \delta_j$ with $\delta_j \overset{\text{iid}}{\sim} \pi$ on $\{-\Delta,\dots,\Delta\}$, $\Delta = 3$ frames $\ge \varepsilon_{\max}$, $\pi$ triangular. With per-frame Bernoulli emissions $y_\tau$ and disjoint windows $W_j = [\hat b_j - \Delta,\, \hat b_j + \Delta]$ (guaranteed by minimum inter-break distance; ties broken by window truncation),

$$\mathcal{L}_{\mathrm{lom}} = -\sum_{j=1}^{J} \log \sum_{\delta = -\Delta}^{\Delta} \pi(\delta)\; y_{\hat b_j + \delta} \!\!\prod_{\substack{\tau \in W_j \\ \tau \ne \hat b_j + \delta}}\!\! (1 - y_\tau)
\;-\; w_-\!\!\sum_{\tau \notin \cup_j W_j}\!\! m_\tau \log (1 - y_\tau). \tag{3}$$

Equation (3) is the exact marginal likelihood of the noisy annotation under the jitter model — the model is never penalized for placing the break anywhere inside the tolerance window, which is what (1) says we actually know. It involves no sequence and no alignment; each window is an independent local latent variable.

### 4.3 Boundary decoding (standalone use)

$$\hat{B} = \{ t : y_t \ge \theta,\;\; y_t = \max_{|t'-t| \le \nu} y_{t'} \}, \qquad \theta = 0.45,\; \nu = 3 \text{ frames (NMS)}.$$

In the full pipeline Model 1 is **not** thresholded; its logits enter the decoder as boundary potentials (§6).

## 5. Model 2: syllable classifier and phonetic kernel

### 5.1 Segment variant (report architecture, retained)

For a span $[s, e)$ with frames $\tilde x_{s:e}$, a 2-layer LSTM (hidden $n_h = 144$, dropout 0.5 between layers) runs

$$\begin{aligned}
i_\tau &= \sigma(W_{xi}\tilde x_\tau + W_{hi} h_{\tau-1} + b_i), &
f_\tau &= \sigma(W_{xf}\tilde x_\tau + W_{hf} h_{\tau-1} + b_f), \\
g_\tau &= \tanh(W_{xg}\tilde x_\tau + W_{hg} h_{\tau-1} + b_g), &
o_\tau &= \sigma(W_{xo}\tilde x_\tau + W_{ho} h_{\tau-1} + b_o), \\
c_\tau &= f_\tau \odot c_{\tau-1} + i_\tau \odot g_\tau, &
h_\tau &= o_\tau \odot \tanh(c_\tau),
\end{aligned}$$

and classifies from the final hidden state of the top layer:

$$p_{\mathrm{M2}}(u \mid \tilde x_{s:e}) = \mathrm{softmax}\big(W_c\, h^{(2)}_{e-1} + b_c\big)_u, \qquad W_c \in \mathbb{R}^{110 \times n_h}.$$

Parameter names (`lstm`, `fc`) and class order are frozen to load the legacy checkpoint (test acc. 0.536) as initialization.

### 5.2 Frame variant (new; enables exact decoding)

A per-frame posterior over the same classes,

$$p_{\mathrm{f}}(u \mid \tilde x_t) = \mathrm{softmax}\big(W_f\, \phi_f(\tilde x_t) + b_f\big)_u,$$

with $\phi_f$ a 1-hidden-layer MLP (256, GELU) or identity. Trained on frames inside labeled segments (target = the segment's token; frames inside $\langle\mathrm{noise}\rangle$ spans excluded; $\langle\mathrm{sil}\rangle$ is an ordinary class), with the phonetic soft targets of §5.3. Its role: additive segment scores $\sum_{t \in [s,e)} \log p_{\mathrm{f}}(u \mid \tilde x_t)$ decompose over frames, allowing the semi-Markov DP of §6 to search **all** segmentations exactly with cumulative sums; the LSTM variant then rescores finalists.

### 5.3 Phonetic kernel $k$ on $\mathcal{T}$

Decompose each non-silence token by its romaji as $\delta(u) = (c(u), \gamma(u), v(u))$: consonant $c \in \mathcal{C} \cup \{\varnothing\}$, palatal glide $\gamma \in \{0,1\}$ (*kya, sho*, …), vowel $v \in \{a,i,u,e,o,\varnothing\}$ ($\varnothing$ for the moraic nasal *n*). Consonant features: $\mathrm{voice}(c) \in \{0,1\}$, $\mathrm{place}(c) \in$ {labial, alveolar, palatal, velar, glottal}, $\mathrm{manner}(c) \in$ {plosive, fricative, affricate, nasal, approximant}. Vowel coordinates $\varphi(v) = (\text{height}, \text{backness})$: $a = (0, 0.5)$, $i = (1, 0)$, $u = (1, 1)$, $e = (0.5, 0)$, $o = (0.5, 1)$.

$$k_C(c, c') = \lambda_v \mathbb{1}[\mathrm{voice}{=}] + \lambda_p \mathbb{1}[\mathrm{place}{=}] + \lambda_m \mathbb{1}[\mathrm{manner}{=}], \quad (\lambda_v, \lambda_p, \lambda_m) = (0.2, 0.4, 0.4),$$

$$k_C(\varnothing,\varnothing) = 1, \quad k_C(\varnothing, c) = 0.15;$$

$$k_V(v, v') = 1 - \tfrac{1}{2}\lVert \varphi(v) - \varphi(v') \rVert_1, \qquad k_V(\varnothing,\varnothing) = 1,\; k_V(\varnothing, v) = 0;$$

$$k(u, u') = \mu_C\, k_C + \mu_V\, k_V + \mu_\gamma \mathbb{1}[\gamma = \gamma'], \qquad (\mu_C, \mu_V, \mu_\gamma) = (0.45, 0.45, 0.10),$$

with overrides: $k(u,u) = 1$; $k(\langle\mathrm{sil}\rangle, u) = \mathbb{1}[u = \langle\mathrm{sil}\rangle]$; modern-Japanese homophone pairs {(*wo*,*o*), (*di*,*ji*), (*du*,*zu*)} set to 0.95. Worked values (matching the report's confusion structure): $k(\text{ka},\text{ga}) = 0.45{\cdot}0.8 + 0.45 + 0.1 = 0.91$ (voicing bit only); $k(\text{ka},\text{ta}) = 0.82$ (place differs); $k(\text{ka},\text{ki}) = 0.66$ (vowel moves); $k(\text{ka},\text{n}) = 0.10$. The full $110{\times}110$ PSD-projected matrix $K$ is precomputed once ($K \leftarrow K - \lambda_{\min}(K) I$ then renormalized, if needed, to guarantee positive semidefiniteness).

**One kernel, three uses:**

1. **Partial-credit loss** (both classifier variants). Soft targets with sharpening power $p = 4$, mass $\alpha = 0.1$:
$$q(u' \mid u) = (1-\alpha)\mathbb{1}[u'{=}u] + \alpha \frac{k(u,u')^{p}\,\mathbb{1}[u' \ne u]}{\sum_{u'' \ne u} k(u,u'')^{p}}, \qquad
\mathcal{L}_{\mathrm{cls}} = -\sum_{u'} q(u' \mid u^\star) \log p(u' \mid \cdot).$$
2. **Soft contrastive negatives** (§9): repulsion between classes $u, u'$ down-weighted by $(1 - k(u,u'))$.
3. **Graded evaluation**: the partial-credit metric of §10.

## 6. Stage D: semi-Markov segmental decoder (textless)

The integration layer. A complete labeled segmentation $\sigma$ (in frames: $\sigma_k = (t_k, s_k, e_k)$, $s_{k+1} = e_k$) is scored by

$$S(\sigma) = \sum_{k=1}^{K} \Big[
\underbrace{\lambda_c \sum_{t=s_k}^{e_k-1} \log p_{\mathrm{f}}(t_k \mid \tilde x_t)}_{\text{acoustic evidence (§5.2)}}
+ \underbrace{\lambda_d \log P_{\mathrm{dur}}(e_k - s_k \mid t_k)}_{\text{duration prior}}
+ \underbrace{\lambda_b\, \beta_{s_k}}_{\text{boundary evidence (§4)}}
+ \underbrace{\lambda_\ell \log A(t_k \mid t_{k-1})}_{\text{token bigram (optional)}}
\Big]. \tag{4}$$

**Duration model.** Per class, shifted negative binomial $d - 1 \sim \mathrm{NB}(r_u, p_u)$:

$$P_{\mathrm{dur}}(d \mid u) = \binom{d-2+r_u}{d-1} (1-p_u)^{d-1} p_u^{r_u}, \qquad d \ge 1,$$

fitted by method of moments from cleaned labels: with $m_u = \bar d_u - 1$, $s_u^2 = \widehat{\mathrm{Var}}(d_u)$, $p_u = m_u / s_u^2$, $r_u = m_u^2/(s_u^2 - m_u)$ (fallback to a pooled geometric when $s_u^2 \le m_u$); classes with $< 30$ observations shrink to the pooled estimate. $\langle\mathrm{sil}\rangle$ gets a heavy-tailed mixture (short gaps vs. instrumental sections): $P_{\mathrm{dur}}(\cdot \mid \langle\mathrm{sil}\rangle) = \omega\,\mathrm{NB}(r_1,p_1) + (1-\omega)\,\mathrm{NB}(r_2,p_2)$, EM-fitted.

**Token bigram.** $A(u' \mid u)$ is an add-$k$ (or Kneser–Ney) smoothed bigram over the training label sequences. It is a prior over $\mathcal{T}$-strings learned offline — **not** a transcript; decoding remains textless. $\lambda_\ell = 0$ is a valid (pure-acoustic) configuration and is the reporting default until the LM's effect is ablated.

**Exact Viterbi DP.** With $D_{\max} = 60$ frames (1.2 s; $\langle\mathrm{sil}\rangle$ exempt via chunking at long low-energy runs), cumulative sums $C_u(t) = \sum_{\tau \le t} \log p_{\mathrm{f}}(u \mid \tilde x_\tau)$ make each span score $O(1)$. Define $V(e, u)$ = best score of any segmentation of $[1, e)$ whose last segment has label $u$:

$$V(0, u) = \mathbb{1}[u = \mathrm{BOS}] \cdot 0,$$

$$V(e, u) = \max_{1 \le d \le D_{\max}} \max_{u' \in \mathcal{T}} \Big[ V(e-d, u') + \lambda_\ell \log A(u \mid u') + \lambda_d \log P_{\mathrm{dur}}(d \mid u)
+ \lambda_c \big(C_u(e-1) - C_u(e-d-1)\big) + \lambda_b\, \beta_{e-d} \Big], \tag{5}$$

$$S^\star = \max_u V(T, u), \qquad \sigma^\star \text{ by backpointers}.$$

Complexity $O(T \cdot D_{\max} \cdot |\mathcal{T}|^2)$ naive; $O(T \cdot D_{\max} \cdot |\mathcal{T}|)$ with $\lambda_\ell = 0$; with a per-span top-$K$ shortlist from $C_u$ ($K = 8$), $O(T \cdot D_{\max} \cdot K^2)$. **Second pass:** the $N$-best segmentations ($N = 4$, extracted by keeping $N$ backpointers) are rescored with the segment LSTM $p_{\mathrm{M2}}$ replacing the frame term, and the best rescored hypothesis is emitted.

**Confidence.** The semi-Markov forward–backward (replace max by logsumexp in (5) for $\alpha$; symmetric $\beta$-recursion) yields per-segment posteriors

$$c_k = p\big(\text{segment } (t_k, s_k, e_k) \in \sigma \mid \tilde x_{1:T}\big)
= \frac{\alpha(s_k, \cdot) \cdot [\text{span factors}] \cdot \beta(e_k, t_k)}{Z},$$

used for pseudo-label filtering (§9) and QA flags (§8).

**Degenerations (interpretability).** $\lambda_d = \lambda_\ell = 0$, $\lambda_b \to \infty$ with thresholded $\beta$: recovers the report's two-stage pipeline (Model-1 breaks, then Model-2 per span). $\lambda_c = \lambda_\ell = 0$: pure boundary+duration segmenter. Each term of (4) is inspectable per decoded segment — the decoder is a scoring identity, not a black box.

## 7. Generative counterpart: sticky HDP-HMM / HSMM (unsupervised)

Used with **no access to token labels or sequences**: (i) acoustic-unit discovery (`kashi discover`), (ii) the unsupervised boundary source for dataset cleaning (§8), (iii) posterior boundary uncertainty.

### 7.1 Generative model

Emissions $x'_t = \mathrm{PCA}_{d'}(g_w(x_t)) \in \mathbb{R}^{d'}$, $d' = 48$ (whitened). With truncation $L = 120$ (weak-limit),

$$\begin{aligned}
\beta &\sim \mathrm{Dir}(\gamma/L, \dots, \gamma/L), \\
\pi_j &\sim \mathrm{Dir}\big(\alpha\beta_1, \dots, \alpha\beta_j + \kappa, \dots, \alpha\beta_L\big), \qquad j = 1,\dots,L,\\
\theta_j = (\mu_j, \mathrm{diag}(\varsigma_j^2)) &\sim \mathrm{NIG}(\mu_0, \lambda_0, a_0, b_0) \text{ per dimension}, \\
z_t \mid z_{t-1} &\sim \pi_{z_{t-1}}, \qquad x'_t \mid z_t \sim \mathcal{N}(\mu_{z_t}, \mathrm{diag}(\varsigma^2_{z_t})).
\end{aligned}$$

$\kappa$ is the sticky self-transition mass ($\rho = \kappa/(\alpha+\kappa) \approx 0.95$ expected self-transition, i.e. expected dwell ≈ 20 frames = 400 ms ≈ mora scale). Boundaries are the **derived** events $\{t : z_t \ne z_{t-1}\}$; no break variable exists, hence no class imbalance exists.

### 7.2 Blocked Gibbs sweep

1. **Path (FFBS).** Forward: $\alpha_t(j) \propto \mathcal{N}(x'_t \mid \theta_j) \sum_{i} \alpha_{t-1}(i)\, \pi_i(j)$. Backward sampling: $z_T \sim \alpha_T(\cdot)$; $z_t \sim \alpha_t(\cdot)\, \pi_{(\cdot)}(z_{t+1})$. One joint draw of $z_{1:T}$ per sweep.
2. **Emissions.** Per state $j$, per dimension: with $n_j$ assigned frames, mean $\bar x_j$, scatter $S_j$: $\lambda_n = \lambda_0 + n_j$, $\mu_n = (\lambda_0\mu_0 + n_j\bar x_j)/\lambda_n$, $a_n = a_0 + n_j/2$, $b_n = b_0 + \tfrac12 S_j + \tfrac{\lambda_0 n_j}{2\lambda_n}(\bar x_j - \mu_0)^2$; draw $\varsigma^2 \sim \mathrm{InvGamma}(a_n, b_n)$, $\mu \sim \mathcal{N}(\mu_n, \varsigma^2/\lambda_n)$.
3. **Transitions.** $\pi_j \sim \mathrm{Dir}(\alpha\beta + \kappa\delta_j + n_{j\cdot})$ with transition counts $n_{jk}$.
4. **Global weights.** Table counts by the Chinese-restaurant sampler $m_{jk} = \sum_{i=1}^{n_{jk}} \mathrm{Bern}\big(\tfrac{\omega_{jk}}{\omega_{jk} + i - 1}\big)$, $\omega_{jk} = \alpha\beta_k + \kappa\mathbb{1}[j{=}k]$; sticky override correction $\bar m_{jj} = m_{jj} - w_j$, $w_j \sim \mathrm{Binom}\big(m_{jj}, \tfrac{\rho}{\rho + \beta_j(1-\rho)}\big)$ (Fox et al. 2011); then $\beta \sim \mathrm{Dir}(\gamma/L + \bar m_{\cdot 1}, \dots, \gamma/L + \bar m_{\cdot L})$.

Defaults: $\gamma = 4$, $\alpha = 4$, $\kappa$ set from $\rho = 0.95$, 30 sweeps after 10 burn-in.

### 7.3 Boundary posterior and HSMM upgrade

From $S$ post-burn-in samples $z^{(s)}$, the boundary probability and location dispersion are

$$\hat p_t = \tfrac{1}{S}\textstyle\sum_s \mathbb{1}[z_t^{(s)} \ne z_{t-1}^{(s)}], \qquad
\varsigma^{\mathrm{bnd}}_m = \mathrm{std}_s(\text{position of matched boundary}) \cdot h \text{ [ms]}.$$

**HSMM:** replace geometric dwell with $d \sim \mathrm{NB}(r, p)$ per state; the forward message over segment ends becomes

$$\alpha_t(j) = \sum_{d=1}^{D_{\max}} \Big[\prod_{\tau=t-d+1}^{t} \mathcal{N}(x'_\tau \mid \theta_j)\Big] P_{\mathrm{dur}}(d \mid j) \sum_{i \ne j} \alpha_{t-d}(i)\, \tilde A(i,j),$$

with backward sampling of $(z, d)$ pairs; complexity $O(T D_{\max} L^2)$, reduced to $O(T D_{\max} L)$ under the sticky/uniform-base factorization. HSMM is adopted iff boundary residuals under the HMM look non-geometric (P6 gate in the roadmap).

## 8. Dataset cleaning without forced alignment

Fixes (1) and unlabeled breaths using only **sequence-free, local** operations. The token sequence is never used to infer times (Remark 1); token identities are only carried along.

### 8.1 Candidate boundaries (unsupervised)

Union of: (i) HDP-HMM/HSMM posterior boundaries $\{(b_m, \hat p_m, \varsigma^{\mathrm{bnd}}_m)\}$ with $\hat p_m \ge 0.3$ (§7); (ii) spectral-flux onsets: $o_t = \sum_f \max(0, |X_{t,f}| - |X_{t-1,f}|)$, peak-picked after per-band normalization; (iii) voicing transitions: extrema of $|v_t - v_{t-1}|$. Merge within 1 frame; candidate confidence = max of source scores.

### 8.2 Monotone boundary snapping

Labeled internal boundaries $\hat u_1 < \dots < \hat u_J$ (segment ends, frames); candidates $b_1 < \dots < b_M$. Feasible matches: $|b_m - \hat u_j| \le \Delta_{\max} = 5$ frames (100 ms). Order-preserving partial matching by the alignment DP

$$D(j, m) = \min\big\{ D(j{-}1, m{-}1) + c(j, m),\;\; D(j{-}1, m) + c_{\mathrm{miss}},\;\; D(j, m{-}1) \big\},
\qquad c(j,m) = \frac{|b_m - \hat u_j|}{\Delta_{\max}} - \eta\, \hat p_m,$$

with $c_{\mathrm{miss}} = 1.2$, $\eta = 0.5$; matches with $c(j,m) > c_{\mathrm{miss}}$ forbidden. Output per boundary: snapped time $u'_j = b_{m(j)}$ (or unchanged if unmatched), $\texttt{moved\_ms}_j = (u'_j - \hat u_j)h$, uncertainty $\varsigma^{\mathrm{bnd}}_{m(j)}$, and $\texttt{flag}_j \in$ {unmatched, large-shift}. This is matching between two **event sets** in time with moves bounded by 100 ms; no transcript, no token-conditioned path, no global warp — categorically not forced alignment.

### 8.3 Breath/noise tagging and QA flags

Inside every labeled $\langle\mathrm{sil}\rangle$ span, tag maximal runs

$$\langle\mathrm{noise}\rangle: \quad \{t : e_t \ge \theta_e \,\wedge\, v_t < \theta_v\} \text{ of length} \ge 3 \text{ frames (60 ms)},
\qquad \theta_e = -35\,\mathrm{dB},\; \theta_v = 0.35$$

(energetic but aperiodic = breath/fricative noise; energetic **and** periodic runs are flagged `missed-vocal` instead — likely mislabeled lyrics). $\langle\mathrm{noise}\rangle$ is a dataset annotation only, never a decoder class; tagged frames are excluded from all training losses. Independent token-level QA (no timing involved): for each snapped segment, if $p_{\mathrm{M2}}(t_k \mid \tilde x_{s'_k:e'_k})$ is in the corpus' bottom $q = 5\%$ for its class, flag `label-suspect`. Per-song QA gate: quarantine iff mean|moved| > 60 ms, or > 15% rows flagged, or candidate-recall < 0.6. Output: `clean_v2` label version + `realign_report.csv`; one retrain-and-repeat iteration (v2 → v3) is accepted only if the gold metrics (§10) improve. **Limits (by design):** snapping corrects jitter ≤ 100 ms; wrong/missing tokens and off-script singing are flagged, never silently edited.

## 9. Learning from unlabeled audio

### 9.1 Temporal InfoNCE (fully unsupervised)

Anchors $a_i = g_w(x_{t_i})$, positives $p_i = g_w(x_{t_i + \delta_i})$, $\delta_i \sim \mathrm{U}\{1,2,3\}$, in-batch negatives, temperature $\tau_c = 0.1$:

$$\mathcal{L}_{\mathrm{nce}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(a_i^\top p_i / \tau_c)}{\sum_{j=1}^{B} \exp(a_i^\top p_j / \tau_c)}.$$

Rationale: frames 20–60 ms apart almost surely share phonetic content but differ in local noise/phase ⇒ $g_w$ learns nuisance-invariant, content-sensitive similarity. Any encoded audio (labeled or scraped) contributes.

### 9.2 Supervised contrastive with soft phonetic negatives

With token labels $u_i$ on segment-pooled embeddings, positives $P(i) = \{j : u_j = u_i\}$ and kernel-weighted repulsion

$$\mathcal{L}_{\mathrm{sup}} = -\frac{1}{B}\sum_i \frac{1}{|P(i)|}\sum_{p \in P(i)} \log
\frac{\exp(a_i^\top a_p/\tau_c)}{\sum_{j \ne i} \big(1 - k(u_i, u_j)\big) \exp(a_i^\top a_j/\tau_c) + \sum_{p' \in P(i)} \exp(a_i^\top a_{p'}/\tau_c)},$$

so *ka* and *ga* ($k = 0.91$) are barely repelled while *ka* and *n* ($k = 0.10$) repel almost fully — the embedding geometry is asked to mirror phonetic geometry instead of fighting it.

### 9.3 Covers (audio-to-audio; later phase)

For two renditions $a, a''$ of one song: DTW on the cosine-distance matrix of frame features yields a monotone **audio-audio** warp $W^\star = \arg\min_W \sum_{(t,t') \in W} \big(1 - \cos(g_w(x_t), g_w(x''_{t'}))\big)$ (Sakoe–Chiba band ±5 s). Matched frame pairs are positives (same syllable content, different singer/timbre). No transcript is involved; this is signal-to-signal correspondence, orthogonal to Remark 1.

### 9.4 Pseudo-labeling (self-training)

Decode unlabeled songs with §6; accept segments with confidence $c_k \ge \theta_c = 0.9$ into a weak-label pool; retrain $p_{\mathrm{f}}, p_{\mathrm{M2}}$ (and optionally Model 1 from decoded boundaries) on labeled + weak data with weight $w_{\mathrm{weak}} = 0.3$; re-evaluate; stop after two non-improving rounds. This is approximate EM on the decoder's own model, entirely textless.

### 9.5 Remark: sequence-marginalized frame training (optional, flagged)

The frame posterior $p_{\mathrm{f}}$ can alternatively be trained from $(x_{1:T}, t_{1:K})$ pairs by marginalizing over all monotone label-to-frame assignments (CTC-style forward–backward on the **training loss**). This uses the labeled sequence exactly as any supervised loss does, never consumes or produces timestamps, and at inference nothing changes (the decoder of §6 remains textless). It is strictly an alternative to trusting v2 timestamps during training and remains **off by default**; it is listed because it is the only known way to train $p_{\mathrm{f}}$ that is provably independent of residual timing noise. If the latent-offset loss (3) + v2 labels suffice on gold metrics, this option stays off.

## 10. Evaluation metrics (formal)

Let gold segments $\{(t^\star_k, s^\star_k, e^\star_k)\}$ and predictions $\{(\hat t_k, \hat s_k, \hat e_k)\}$, silences stripped and excluded rows dropped unless stated.

1. **Boundary F1@τ**, $\tau \in \{20, 50\}$ ms: greedy monotone 1–1 matching of boundary sets within $\tau$; $F_1 = 2PR/(P{+}R)$. Also mean$|\Delta t|$ over matched pairs.
2. **SER**: $\mathrm{Lev}\big((\hat t_k)_k, (t^\star_k)_k\big) / K^\star$ (token-level Levenshtein; the end-to-end headline).
3. **Timed-token F1**: a predicted segment is correct iff $\hat t = t^\star$ for its matched gold segment and $|\hat s - s^\star| \le 50$ ms; matching greedy by start time.
4. **Partial credit**: $\mathrm{PC} = \mathbb{E}[\, k(\hat t, t^\star) \,]$ over matched pairs (the report's §7.2 made concrete).
5. **Noise-span P/R**: prediction and gold $\langle\mathrm{noise}\rangle$ spans match iff $\mathrm{IoU} \ge 0.3$.

Model selection uses Boundary F1@50 (Model 1), $\mathcal{L}_{\mathrm{cls}}$/accuracy (Model 2), SER + timed-token F1 (pipeline); promotion gates on the frozen paper test split; gold windows on test songs are reporting-only.

## 11. Defaults and complexity

| Component | Key defaults | Train cost (6 GB GPU) | Inference cost / 4-min song |
|---|---|---|---|
| Separator $V$ | mel-band roformer ckpt | none (frozen) | ~1–3 min (dominant) |
| Backbone $\Phi$ | JA base ckpt, frozen, 20 ms | none | ~10 s |
| Projection $g_w$ | 768→256→128, $\tau_c{=}0.1$ | minutes (cached feats) | negligible |
| Voicing $v_t$ | autocorr, [70, 500] Hz | — | $O(T L_w \log L_w)$, seconds |
| Model 1 | $d_m{=}256$, $L{=}2$, $H{=}8$, $W_a{=}100$; $\mathcal{L}_{\mathrm{lom}}$, $\Delta{=}3$ | tens of min | $O(T W_a d_m)$, seconds |
| Model 2 (LSTM) | $n_h{=}144{\times}2$, dropout 0.5; $\alpha{=}0.1$, $p{=}4$ | tens of min | rescoring only |
| Model 2f (frame) | linear/MLP-256 head | minutes | $O(T d\,\lvert\mathcal{T}\rvert)$ |
| Decoder $D$ | $D_{\max}{=}60$, $K{=}8$, $N{=}4$; $\lambda_c{=}1$, $\lambda_d{=}0.5$, $\lambda_b{=}0.5$, $\lambda_\ell{\in}\{0, 0.3\}$ | λ's by grid on dev | $O(T D_{\max} K^2)$, seconds |
| HDP-HMM | $L{=}120$, $d'{=}48$, $\rho{=}0.95$, 30+10 sweeps | — (CPU ok) | $O(S\,T L D_{\max})$ HSMM, minutes |
| Snapping | $\Delta_{\max}{=}5$ fr, $c_{\mathrm{miss}}{=}1.2$, $\eta{=}0.5$ | — | $O(JM)$ DP, ms |

**Two-stage compatibility.** The report's pipeline is the decoder degeneration $\lambda_d = \lambda_\ell = 0$ with hard Model-1 boundaries; it is retained as a registry configuration and as the P1 baseline. All weights λ and thresholds above are config keys, not code constants.
