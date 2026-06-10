# Thesis architecture diagrams (Mermaid source for manual drawing)

Reference for the Chapter 3 architecture figures: the monolithic **baselines** and the proposed
**semantic-group hybrid**. Every layer and its width is listed (dims taken from
`src/models/models.py`). These are not compiled into the thesis — paste a block into any Mermaid
renderer (GitHub, VS Code "Markdown Preview Mermaid Support", or <https://mermaid.live>) to view,
then draw by hand and export to:

- `documents/thesis/Figure/baseline_architecture.png` → Diagram 1
- `documents/thesis/Figure/hybrid_architecture.png`   → Diagram 2

Conventions: `T` = number of frames; group dims are gaze 8, eye 280, face 340, head pose 46,
action units 35. Every classifier/encoder hidden layer is **128 wide** (the shared hidden width of
the temporal architectures); the five group embeddings are **64-d** and the I3D embedding is
**128-d** (concatenation = 5x64 + 128 = 448). All dropouts are 0.3 except inside the temporal
encoders (0.2).

---

## Diagram 1 — Baseline (monolithic) architectures

### 1a. OpenFace baselines (one encoder over the full `T x 709` sequence)

`openface_mlp`
```mermaid
flowchart TB
    A["input: T x 709"] --> B["Flatten: T*709"] --> C["Linear -> 256"] --> D["ReLU"] --> E["Dropout 0.3"] --> F["Linear 256 -> 128"] --> G["ReLU"] --> H["Dropout 0.3"] --> I["Linear 128 -> 4"] --> J["logits (4)"]
```

`openface_tcn` (temporal_cnn)
```mermaid
flowchart TB
    A["input: T x 709"] --> B["TemporalBlock 1: 2x Conv1d, 256 ch, kernel 3, dilation 1<br/>(weight-norm, ReLU, Dropout 0.2) + residual"]
    B --> C["TemporalBlock 2: 2x Conv1d, 128 ch, kernel 3, dilation 2 + residual"]
    C --> D["TemporalBlock 3: 2x Conv1d, 128 ch, kernel 3, dilation 4 + residual"]
    D --> E["AdaptiveAvgPool1d(1) -> 128"]
    E --> F["Dropout 0.3"] --> G["Linear 128 -> 128"] --> H["ReLU"] --> I["Dropout 0.3"] --> J["Linear 128 -> 4"] --> K["logits (4)"]
```

`openface_lstm`
```mermaid
flowchart TB
    A["input: T x 709"] --> B["LSTM: 2 layers, hidden 256, dropout 0.3"] --> C["last-layer hidden state: 256"]
    C --> D["Dropout 0.3"] --> E["Linear 256 -> 128"] --> F["ReLU"] --> G["Dropout 0.3"] --> H["Linear 128 -> 4"] --> I["logits (4)"]
```

`openface_transformer`
```mermaid
flowchart TB
    A["input: T x 709"] --> B["Linear 709 -> 128"] --> C["+ sinusoidal positional encoding"]
    C --> D["2x TransformerEncoderLayer<br/>d_model 128, heads 4, feed-forward 256, GELU, dropout 0.2"]
    D --> E["mean-pool over time -> 128"] --> F["LayerNorm(128)"]
    F --> G["Dropout 0.3"] --> H["Linear 128 -> 128"] --> I["ReLU"] --> J["Dropout 0.3"] --> K["Linear 128 -> 4"] --> L["logits (4)"]
```

### 1b. I3D baseline (`i3d_mlp`) — MLP on the 1024-d clip vector

```mermaid
flowchart TB
    A["input: I3D clip vector, 1024-d<br/>(tiled to T x 1024 by the loader)"] --> B["mean-pool over time -> 1024"]
    B --> C["Linear 1024 -> 256"] --> D["ReLU"] --> E["Dropout 0.3"] --> F["Linear 256 -> 128"] --> G["ReLU"] --> H["Dropout 0.3"] --> I["Linear 128 -> 4"] --> J["logits (4)"]
```

---

## Diagram 2 — Semantic-group hybrid architecture

Each OpenFace group and the I3D stream is encoded independently into a 64-d embedding; the six
embeddings are concatenated and classified by a shared MLP; each stream also feeds an auxiliary
head. OpenFace-only variant: drop the I3D row, so the concatenation is `5 x 64 = 320` and the
fusion MLP is `320 -> 128 -> 4`.

### 2a. Overall data flow

```mermaid
flowchart LR
    GZ["gaze: T x 8"] --> GZe["Group encoder -> 64-d<br/>(TCN | Transformer | LSTM; see 2b)"] --> GZm["64-d"]
    EY["eye: T x 280"] --> EYe["Group encoder -> 64-d"] --> EYm["64-d"]
    FC["face: T x 340"] --> FCe["Group encoder -> 64-d"] --> FCm["64-d"]
    HP["head pose: T x 46"] --> HPe["Group encoder -> 64-d"] --> HPm["64-d"]
    AU["action units: T x 35"] --> AUe["Group encoder -> 64-d"] --> AUm["64-d"]
    I3["I3D: 1024-d clip vector"] --> I3e["mean-pool over time -> 1024<br/>Linear 1024 -> 256, ReLU, Dropout 0.3<br/>Linear 256 -> 128, ReLU<br/>LayerNorm(128)"] --> I3m["128-d"]

    GZm --> CAT["Concatenate: 5 x 64 + 128 = 448"]
    EYm --> CAT
    FCm --> CAT
    HPm --> CAT
    AUm --> CAT
    I3m --> CAT
    CAT --> F1["Linear 448 -> 128"] --> F2["ReLU"] --> F3["Dropout 0.3"] --> F4["Linear 128 -> 4"] --> OUT["main logits (4)"]

    GZm -.-> AUX["Auxiliary head per stream:<br/>Dropout 0.3, Linear 64 -> 4 (groups) / 128 -> 4 (I3D)<br/>aux loss = cross-entropy, weight 0.2<br/>total loss = main + 0.2 * mean(aux)"]
    EYm -.-> AUX
    FCm -.-> AUX
    HPm -.-> AUX
    AUm -.-> AUX
    I3m -.-> AUX
```

### 2b. Group encoder options (each maps `T x D_group` to a 64-d embedding)

A group's encoder is one of these three; `D_group` is that group's width (8 / 280 / 340 / 46 / 35).

```mermaid
flowchart TB
    subgraph OptA["Option A: TCN"]
      direction TB
      a1["input: T x D_group"] --> a2["TemporalBlock 1: 2x Conv1d, 64 ch, kernel 3, dilation 1<br/>(weight-norm, ReLU, Dropout 0.2) + residual"] --> a3["TemporalBlock 2: 2x Conv1d, 64 ch, kernel 3, dilation 2 + residual"] --> a4["mean-pool over time -> 64"] --> a5["LayerNorm(64)"]
    end
    subgraph OptB["Option B: Transformer"]
      direction TB
      b1["input: T x D_group"] --> b2["Linear D_group -> 64"] --> b3["+ sinusoidal positional encoding"] --> b4["2x TransformerEncoderLayer<br/>d_model 64, heads 4, feed-forward 128, GELU, dropout 0.2"] --> b5["mean-pool over time -> 64"] --> b6["LayerNorm(64)"]
    end
    subgraph OptC["Option C: LSTM"]
      direction TB
      c1["input: T x D_group"] --> c2["LSTM: 2 layers, hidden 64, dropout 0.2"] --> c3["mean-pool over time -> 64"] --> c4["LayerNorm(64)"]
    end
```
