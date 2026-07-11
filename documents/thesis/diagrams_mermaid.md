# Thesis architecture diagrams (Mermaid source for manual drawing)

Reference for the Chapter 3 architecture figures: the single-encoder **baselines** and the proposed
**feature-group hybrid**. Every layer and its width is listed (dims taken from
`src/models/models.py`). These are not compiled into the thesis — paste a block into any Mermaid
renderer (GitHub, VS Code "Markdown Preview Mermaid Support", or <https://mermaid.live>) to view,
then draw by hand and export to:

- `documents/thesis/Figure/baseline_architecture.png` → Diagram 1
- `documents/thesis/Figure/hybrid_architecture.png`   → Diagram 2

Conventions: `T` = number of frames; group dims are gaze 8, eye 280, face 340, head pose 46,
action units 35. Every classifier/encoder hidden layer is **128 wide** (the shared hidden width of
the temporal architectures); the five group embeddings are **64-d** and the I3D embedding is
**128-d** (concatenation = 5x64 + 128 = 448). All dropouts are 0.3 except inside the temporal
encoders (0.2). Boxes are rounded; `classDef default ... rx:7px,ry:7px` rounds every node.

> **Rendering / export notes (read once):**
> - **Bigger text in boxes:** every block has a `themeVariables.fontSize` config; raise it if text
>   still looks small relative to the box.
> - **Long pipelines that go tiny in the thesis:** most diagrams are a single left-to-right row
>   (`flowchart LR`). The two longest baselines (`openface_tcn`, `openface_transformer`) are wrapped
>   onto **two labelled stage rows**: outer `flowchart TB`, two `subgraph "<stage>" ... direction LR`
>   rows kept stacked by an **invisible** `r1 ~~~ r2` link (renders nothing) — draw the D2→E arrow
>   between the two stage boxes by hand in the editor. (A true node→node arrow across the wrap, e.g.
>   `D2 --> E`, is impossible — any node with a cross-subgraph edge makes dagre/elk drop that row's
>   `direction LR`, verified by rendering, which is why the connector is left to the manual drawing
>   step.) **Export as SVG or PDF (vector), not PNG**, so wide pipelines stay crisp when LaTeX scales
>   them down.
> - **Residual / elementwise operators:** drawn with the math-standard glyphs `⊕` (add) and `⊗`
>   (multiply) as borderless nodes (`:::op`, `font-size: 60px`), instead of a circle wrapping a `+`.

---

## Diagram 0 — Overview figures (Figure 3.1)

The two overview schematics that contrast the paradigms. Capture and export to:

- `documents/thesis/Figure/overview_baseline.png` → Diagram 0a
- `documents/thesis/Figure/overview_hybrid.png`   → Diagram 0b

(These replace the inline TikZ currently used for Figure 3.1 in `Chapter/3_Methodology.tex`.)

### 0a. Baseline (single-encoder) overview — all five baselines, both MLPs included

```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    A["OpenFace sequence\nT x 709"] --> B["Single encoder over the full sequence\nMLP / TCN / LSTM / Transformer"] --> C["Classifier"] --> Z1["Ordinal level\nE0 / E1 / E2 / E3"]
    D["I3D clip vector\n1024"] --> E["MLP"] --> F["Classifier"] --> Z2["Ordinal level\nE0 / E1 / E2 / E3"]
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```

Two independent pipelines, no fusion box (the contrast with 0b). Four OpenFace baselines apply one
encoder (MLP / TCN / LSTM / Transformer) to the full 709-d sequence; the I3D baseline is an MLP on
the pooled 1024-d vector. Both MLP baselines (`openface_mlp`, `i3d_mlp`) are shown.

### 0b. Proposed hybrid overview (auxiliary heads omitted; detailed in Diagram 2a / Figure 3.4)

```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    OF["OpenFace sequence\nT x 709"] --> G1["Gaze encoder\n64"]
    OF --> G2["Eye-landmark encoder\n64"]
    OF --> G3["Face-landmark encoder\n64"]
    OF --> G4["Head-pose encoder\n64"]
    OF --> G5["Action-unit encoder\n64"]
    IV["I3D clip vector\n1024"] --> I3["I3D MLP\n128"]
    G1 --> CAT["Concatenate\n5x64 + 128 = 448"]
    G2 --> CAT
    G3 --> CAT
    G4 --> CAT
    G5 --> CAT
    I3 --> CAT
    CAT --> FUS["Fusion MLP\n448 -> 128 -> 4"]
    FUS --> Z["Ordinal level\nE0 / E1 / E2 / E3"]
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```

I3D-stream-disabled variant: drop the I3D row, so the concatenation is `5 x 64 = 320` and the fusion
MLP is `320 -> 128 -> 4`.

---

## Diagram 1 — Baseline (single-encoder) architectures

### 1a. OpenFace baselines (one encoder over the full `T x 709` sequence)

`openface_mlp`
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    A["Input\nT x 709"] --> B["Flatten\nT*709"] --> C["Linear\n 256"] --> D["ReLU\nDropout 0.3"]
    D --> E["Linear\n 128"] --> F["ReLU\nDropout 0.3"] --> G["Linear\n 4"]
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```

`openface_tcn` (temporal_cnn)
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart TB
    subgraph r1[" "]
      direction LR
      A["Input\nT x 709"] --> B["TemporalBlock 1\n256 ch, dilation 1"] --> C["TemporalBlock 2\n128 ch, dilation 2"] --> D["TemporalBlock 3\n128 ch, dilation 4"]
    end
    subgraph r2[" "]
      direction LR
      E["AdaptiveAvgPool\n 128"] --> F["Dropout 0.3"] --> G["Linear \n 128"] --> H["ReLU\nDropout 0.3"] --> I["Linear\n 4"]
    end
    r1 ~~~ r2
    style r1 fill:none,stroke:none
    style r2 fill:none,stroke:none
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```
`openface_tcn(subgraph)`
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    subgraph BLK["TemporalBlock (kernel 3)"]
      direction LR
      t0["Input"] --> t1["Conv1d\nweight-norm"]
      t1 --> t2["ReLU\nDropout 0.2"]
      t2 --> t3["Conv1d\nweight-norm"]
      t3 --> t4["ReLU\nDropout 0.2"]
      t4 --> t5["⊕"]:::op
      t0 -- residual --> t5
      t5 --> t6["Output"]
    end
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
    classDef op fill:none,stroke:none,font-size:60px
```

`openface_lstm`
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    A["Input\nT x 709"] --> L1["LSTM layer 1\nhidden 256"] --> LD["Dropout 0.3"] --> L2["LSTM layer 2\nhidden 256"]
    L2 --> D["Dropout 0.3"] --> E["Linear\n 128"] --> F["ReLU\nDropout 0.3"] --> G["Linear\n 4"]
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```

`openface_lstm(subgraph)` — one LSTM cell, unrolled per timestep `t`
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    subgraph CELL["LSTM cell (timestep t)"]
      direction LR
      xt["x_t"] --> F["Forget gate\nf = σ(W_f·[h_(t-1), x_t])"]
      xt --> I["Input gate\ni = σ(W_i·[h_(t-1), x_t])"]
      xt --> G["Candidate\ng = tanh(W_g·[h_(t-1), x_t])"]
      xt --> O["Output gate\no = σ(W_o·[h_(t-1), x_t])"]
      hp["h_(t-1)"] --> F
      hp --> I
      hp --> G
      hp --> O

      cp["C_(t-1)"] --> M1["⊗"]:::op
      F --> M1
      I --> M2["⊗"]:::op
      G --> M2
      M1 --> S["⊕"]:::op
      M2 --> S
      S --> Ct["C_t"]
      Ct --> TH["tanh"]
      TH --> M3["⊗"]:::op
      O --> M3
      M3 --> Ht["h_t"]
    end
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
    classDef op fill:none,stroke:none,font-size:60px
```

`openface_transformer`
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart TB
    subgraph r1[" "]
      direction LR
      A["Input\nT x 709"] --> B["Input projection\nLinear 128"]
      PE["Positional encoding\nsinusoidal 128"] --> ADD["⊕"]:::op
      B --> ADD
      ADD --> D1["EncoderLayer 1\nd 128, heads 4, FF 256"] --> D2["EncoderLayer 2\nd 128, heads 4, FF 256"]
    end
    subgraph r2[" "]
      direction LR
      E["Mean-pool over T\n 128"] --> G["LayerNorm 128 \n Dropout 0.3"] --> H["Linear\n 128"] --> I["ReLU\nDropout 0.3"] --> J["Linear\n 4"]
    end
    r1 ~~~ r2
    style r1 fill:none,stroke:none
    style r2 fill:none,stroke:none
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
    classDef op fill:none,stroke:none,font-size:60px
```

`openface_transformer(subgraph)` — one encoder layer (post-norm)
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart TB
    subgraph r1[" "]
      direction LR
      in["Input\nT x d"] --> MHA["Multi-Head\nself-attention"] --> Dr1["Dropout 0.2"] --> Ad1["⊕"]:::op
      Ad1 --> LN1["LayerNorm"]
      in -- residual --> Ad1
    end
    subgraph r2[" "]
      direction LR
      FF1["Linear\nd -> FF, GELU"] --> Dr2["Dropout 0.2"] --> FF2["Linear\nFF -> d"] --> Dr3["Dropout 0.2"] --> Ad2["⊕"]:::op
      Ad2 --> LN2["LayerNorm"] --> out["Output\nT x d"]
    end
    r1 ~~~ r2
    style r1 fill:none,stroke:none
    style r2 fill:none,stroke:none
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
    classDef op fill:none,stroke:none,font-size:60px

%% Draw these two seam arrows by hand in the editor (cross-row, can't be auto-rendered without
%% collapsing the LR rows): LN1 --> FF1  (main flow)   and   LN1 -- residual --> Ad2  (skip connection)
```

### 1b. I3D baseline (`i3d_mlp`) — MLP on the 1024-d clip vector

```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    A["Input\nI3D clip 1024-d"] --> C["Linear\n1024 -> 256"] --> D["ReLU\nDropout 0.3"]
    D --> E["Linear\n256 -> 128"] --> F["ReLU\nDropout 0.3"] --> G["Linear\n128 -> 4"]
    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```

---

## Diagram 2 — Feature-group hybrid architecture

Each OpenFace group and the I3D stream is encoded independently into a 64-d embedding; the six
embeddings are concatenated and classified by a shared MLP; each stream also feeds an auxiliary
head. OpenFace-only variant: drop the I3D row, so the concatenation is `5 x 64 = 320` and the
fusion MLP is `320 -> 128 -> 4`.

### 2a. Overall data flow

```mermaid
---
config:
  layout: elk
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    GZ["gaze\nT x 8"] --> GZe["Group encoder\n 64\n"]
    EY["eye\nT x 280"] --> EYe["Group encoder\n 64"]
    FC["face\nT x 340"] --> FCe["Group encoder\n 64"]
    HP["head pose\nT x 46"] --> HPe["Group encoder\n 64"]
    AU["action units\nT x 35"] --> AUe["Group encoder\n 64"]
    I3["I3D\n1024"] --> I3e["Linear 128\nLayerNorm"]
    GZe --> CAT["Concatenate\n5x64 + 128 = 448"]
    EYe --> CAT
    FCe --> CAT
    HPe --> CAT
    AUe --> CAT
    I3e --> CAT
    CAT --> F1["Linear\n 128"]
    F1 --> F2["ReLU\nDropout 0.3"]
    F2 --> F3["Linear\n 4"]
    GZe -.-> AUX["Aux head per stream\nLinear 4"]
    EYe -.-> AUX
    FCe -.-> AUX
    HPe -.-> AUX
    AUe -.-> AUX
    I3e -.-> AUX

    classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
```

Total loss = main cross-entropy + `0.2 * mean(aux cross-entropy over the six streams)`.

### 2b. Group encoder options (each maps `T x D_group` to a 64-d embedding)

A group's encoder is one of these three; `D_group` is that group's width (8 / 280 / 340 / 46 / 35).

Option A: TCN
```mermaid
---
config:
  theme: neutral
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    subgraph ENC["Group Encoder Option TCN"]
    direction LR
        a1["Input\nT x D_group"] --> a2["TemporalBlock 1\n64 ch, dilation 1"]
        a2 --> a3["TemporalBlock 2\n64 ch, dilation 2"]
        a3 --> a4["Mean-pool\n64"]
        a4 --> a5["LayerNorm\n64"]
        classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
    end
```

Option B: Transformer (encoder-layer internals in `openface_transformer(subgraph)`, with `d = 64`, `FF = 128`)
```mermaid
---
config:
  theme: neutral
  layout: elk
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    subgraph ENC["Group Encoder Option Transformer"]
        direction LR
        b1["Input\nT x D_group"] --> b2["Input projection\nLinear D_group 64"]
        bpe["Positional encoding\nsinusoidal, 64-d"] --> badd["⊕"]:::op
        b2 --> badd
        badd --> b4a["EncoderLayer 1\nd 64, heads 4, FF 128"]
        b4a --> b4b["EncoderLayer 2\nd 64, heads 4, FF 128"]
        b4b --> b5["Mean-pool over T\n 64"]
        b5 --> b6["LayerNorm\n64"]
        classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
        classDef op fill:none,stroke:none,font-size:60px
        end
```

Option C: LSTM (cell internals in `openface_lstm(subgraph)`, with hidden `64`)
```mermaid
---
config:
  themeVariables:
    fontSize: 20px
  flowchart:
    padding: 4
    nodeSpacing: 30
    rankSpacing: 40
---
flowchart LR
    subgraph ENC["Group Encoder Option LSTM"]
        direction LR
        c1["Input\nT x D_group"] --> cl1["LSTM layer 1\nhidden 64"]
        cl1 --> cld["Dropout 0.2"]
        cld --> cl2["LSTM layer 2\nhidden 64"]
        cl2 --> c3["Mean-pool over T\n-> 64"]
        c3 --> c4["LayerNorm\n64"]
        classDef default fill:#ffffff,stroke:#333,stroke-width:2px,rx:7px,ry:7px
        classDef op fill:none,stroke:none,font-size:60px
        end

```

> TemporalBlock internals (shared by 1a/2b TCNs): `Conv1d (weight-norm) -> ReLU -> Dropout 0.2`
> twice, with a residual connection; kernel 3 throughout. See the detail subgraph in `openface_tcn`.
>
> LSTM/Transformer internals are shared across 1a and 2b — only widths differ (baseline:
> hidden 256 / `d=128`, `FF=256`; group encoder: hidden 64 / `d=64`, `FF=128`). The group LSTM
> mean-pools its per-timestep outputs (the baseline instead takes the last hidden state `h_T`).
