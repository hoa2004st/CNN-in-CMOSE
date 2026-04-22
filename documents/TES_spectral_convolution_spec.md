# Temporal Engagement Signatures (TES) via Spectral Convolution

## 1. Overview and Motivation

### Problem Statement
Current temporal models (Temporal CNN, LSTM, Transformer) treat engagement as a static classification problem over a fixed 300-frame sequence. However, engagement is inherently **oscillatory** and **dynamic**:

- **Engaged students** display stable facial behavior: consistent gaze, minimal head jitter, smooth AU patterns
- **Disengaged students** display frequent state changes: fidgeting, looking away, rapid expression changes
- **Moderately engaged students** show intermediate frequency of behavioral changes

The key insight: **engagement manifests as specific temporal frequency signatures in facial features**.

### Hypothesis
By analyzing facial features in the **frequency domain** (not just time domain), we can:
1. Extract which temporal frequencies correlate with engagement levels
2. Capture engagement-relevant patterns that raw temporal convolution misses
3. Provide interpretable outputs ("students show low-frequency head stability when engaged")

### Expected Contribution
- **Novel**: Spectral analysis rarely applied to facial engagement detection (common in speech/EEG but untested for facial)
- **Interpretable**: Frequency decomposition reveals *how* engagement manifests temporally
- **Potentially superior**: Engagement patterns may be more separable in frequency domain than time domain

---

## 2. Theoretical Foundation

### 2.1 Why Spectral Analysis?

Consider two hypothetical students over 10 seconds (300 frames):

**Student A (Engaged):**
- Head pose: smooth, slow changes (low frequency)
- Gaze direction: stable (very low frequency, high power at DC component)
- Facial AU intensity: steady (low frequency)

**Student B (Disengaged):**
- Head pose: frequent fidgeting, head tilts (high frequency, energy spread)
- Gaze direction: rapid shifts, looking away (mid-high frequency)
- Facial AU intensity: rapid micro-expressions (high frequency)

**Standard temporal models** see sequences of similar values; they miss that **the *rate* of change differs**. Spectral decomposition directly captures this.

### 2.2 Short-Time Fourier Transform (STFT)

We use **STFT** rather than global FFT because:
- Global FFT loses temporal localization (we lose "when" engagement changes)
- STFT preserves time-frequency localization: tells us engagement signature *over the clip*
- Sliding window (e.g., 32-frame window, 50% overlap) creates feature maps

**STFT Formula:**
```
X_stft[t, f] = sum_{n} x[n] * w[n - t] * exp(-2πi*f*n/N)
```
Where:
- `t` = time frame (0 to ~100 windows for 300 total frames)
- `f` = frequency bin (0 to ~16 Hz for engagement; facial changes are slow)
- `x[n]` = input feature (e.g., head pose yaw)
- `w[n]` = window function (Hann window)

**Output**: magnitude spectrogram of shape `(time_windows, frequency_bins)` per feature.

### 2.3 Physics of Engagement in Frequency Space

**Rough frequency bands for facial features:**

| Frequency Band | Phenomenon | Engagement Association |
|---|---|---|
| **0.1 - 0.5 Hz** (period: 2-10 sec) | Overall engagement level shifts | Sustained attention ↔ Disengagement transition |
| **0.5 - 2 Hz** (period: 0.5-2 sec) | Micro-expressions, blink cycles, slow AU changes | Moderate concentration |
| **2 - 5 Hz** (period: 0.2-0.5 sec) | Rapid fidgeting, head jitter, tremor | Disengagement, anxiety |
| **5+ Hz** | Noise, facial muscle tremor | Measurement noise |

**Hypothesis**: Engaged students will have **high power in 0.1-0.5 Hz** (stable long-term); disengaged will have **high power in 2-5 Hz** (fidgeting).

---

## 3. Implementation Architecture

### 3.1 High-Level Pipeline

```
Raw OpenFace (300 frames × 709 features)
    ↓
[NORMALIZATION] (z-score per feature using train stats)
    ↓
[FEATURE GROUPING] (select subset of 709 for spectral analysis)
    ├─ Group 1: Head Pose (3 features: yaw, pitch, roll)
    ├─ Group 2: Gaze (4 features: gaze_0, gaze_1, gaze_angle_x, gaze_angle_y)
    ├─ Group 3: Action Units (17 AU intensities: AU01, AU02, ... AU45)
    └─ Group 4: Eye Landmarks (72 coordinates: landmarks_x, landmarks_y for 36 points)
    ↓
[STFT PER FEATURE] for each of ~100 selected features
    → output: (n_features, n_time_windows, n_freq_bins)
    ↓
[SPECTRAL CONV LAYERS] 
    → Learn filters over frequency dimension and time-frequency patterns
    → Output: (128 channels)
    ↓
[TEMPORAL CONV LAYERS] 
    → Learn temporal dynamics in *frequency space*
    → Output: (256 channels)
    ↓
[ADAPTIVE POOLING + CLASSIFIER]
    → 4-class engagement prediction
    ↓
Logits → Softmax → Output class (0, 1, 2, 3)
```

### 3.2 Architecture Details

#### Phase 1: STFT Feature Extraction

**Input**: `X_train_normalized` shape `(n_samples, 300, 709)` – already z-score normalized

**Process per sample**:
```python
def extract_stft_features(frame_sequence, feature_indices):
    """
    Args:
        frame_sequence: (300, 709) – 300 frames of 709 OpenFace features
        feature_indices: list of selected feature columns to compute STFT on
    
    Returns:
        stft_magnitude: (len(feature_indices), n_time_windows, n_freq_bins)
    """
    selected_features = frame_sequence[:, feature_indices]  # (300, n_selected)
    
    stft_list = []
    for feature_idx in range(selected_features.shape[1]):
        signal = selected_features[:, feature_idx]  # (300,)
        
        # STFT parameters:
        # - nperseg = 32 frames → ~1 sec window (at 30 fps)
        # - noverlap = 16 frames → 50% overlap
        # - nfft = 64 → zero-pad to 64 for frequency resolution
        f, t, Zxx = scipy.signal.stft(
            signal,
            fs=30,  # 30 fps
            nperseg=32,
            noverlap=16,
            nfft=64,
            window='hann'
        )
        # Zxx shape: (33 freq_bins, ~9 time_windows) for 300 frames
        # Magnitude: |Zxx|
        stft_mag = np.abs(Zxx)  # (33, 9)
        stft_list.append(stft_mag)
    
    # Stack: (n_selected, 33, 9)
    return np.array(stft_list)
```

**Key decisions**:
- **nperseg=32**: 32 frames = ~1 second at 30 fps; captures meaningful engagement transitions
- **noverlap=16**: 50% overlap avoids discontinuities
- **nfft=64**: Provides ~0.5 Hz frequency resolution (30 Hz / 64 = 0.47 Hz per bin)
- **Take magnitude only**: Phase is less informative for engagement

**Output shape**: `(n_samples, n_selected_features, n_freq_bins=33, n_time_windows=9)`

**Feature selection** (subset of 709 for efficiency):
```python
SELECTED_FEATURES = {
    'head_pose': ['pose_Tx', 'pose_Ty', 'pose_Rz'],  # 3
    'gaze': ['gaze_0', 'gaze_1', 'gaze_angle_x', 'gaze_angle_y'],  # 4
    'aus': [f'AU{i:02d}_r' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 43]],  # 17 AUs
    'eye_landmarks': [f'x_{i}' for i in range(0, 144, 2)] + [f'y_{i}' for i in range(1, 144, 2)],  # ~72 coords
}
# Total: ~100 features (adjust based on availability in your CSV)
```

---

#### Phase 2: Spectral CNN Architecture

**Goal**: Learn filters over frequency (and time-frequency patterns) that separate engagement levels.

```python
class SpectralConvNet(nn.Module):
    """
    Input: (batch, n_features, n_freq_bins, n_time_windows)
           e.g., (8, 100, 33, 9)
    """
    def __init__(self, n_input_features=100, n_freq_bins=33, num_classes=4):
        super().__init__()
        
        # SPECTRAL BLOCK 1: Learn frequency-domain patterns
        # Kernel shape: (height=n_freq_bins, width=3)
        # This learns how different frequencies contribute
        self.spec_conv1 = nn.Conv2d(
            in_channels=n_input_features,
            out_channels=64,
            kernel_size=(5, 3),  # (freq_kernel=5, time_kernel=3)
            padding=(2, 1),
            stride=(1, 1)
        )
        self.spec_bn1 = nn.BatchNorm2d(64)
        self.spec_relu1 = nn.ReLU(inplace=True)
        
        # Max pool only on time dimension (preserve frequency structure)
        self.spec_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        # SPECTRAL BLOCK 2: Learn joint freq-time patterns
        self.spec_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1)
        )
        self.spec_bn2 = nn.BatchNorm2d(128)
        self.spec_relu2 = nn.ReLU(inplace=True)
        self.spec_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # SPECTRAL BLOCK 3: Aggregate frequency information
        self.spec_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.spec_bn3 = nn.BatchNorm2d(128)
        self.spec_relu3 = nn.ReLU(inplace=True)
        
        # Global average pooling over frequency and time
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(128, 128)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc_dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_features, n_freq_bins, n_time_windows)
        """
        # Spectral conv path
        x = self.spec_conv1(x)  # (batch, 64, 33, 9)
        x = self.spec_bn1(x)
        x = self.spec_relu1(x)
        x = self.spec_pool1(x)  # (batch, 64, 33, 4)
        
        x = self.spec_conv2(x)  # (batch, 128, 33, 4)
        x = self.spec_bn2(x)
        x = self.spec_relu2(x)
        x = self.spec_pool2(x)  # (batch, 128, 16, 2)
        
        x = self.spec_conv3(x)  # (batch, 128, 16, 2)
        x = self.spec_bn3(x)
        x = self.spec_relu3(x)
        
        # Global aggregation
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)
        
        # Classification
        x = self.dropout(x)
        x = self.fc1(x)  # (batch, 128)
        x = self.fc_relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)  # (batch, 4)
        
        return x
```

**Key design choices**:
- **Asymmetric kernels**: `(5, 3)` → prioritize frequency patterns, but also capture time dynamics
- **Pool only on time**: Preserve frequency dimension (engagement signature lives in frequency space)
- **Batch norm**: Helps stabilization; spectral magnitudes can have high variance
- **Global average pool**: Aggregate all frequency-time information without spatial bias

---

### 3.3 Data Pipeline

**Preprocessing modifications** (extends existing pipeline):

```python
class SpectralPreprocessor:
    """
    Converts raw 300 × 709 frame-feature matrices to spectral tensors.
    """
    
    def __init__(self, feature_indices=None, n_fft=64, nperseg=32, noverlap=16):
        self.feature_indices = feature_indices or self._default_indices()
        self.n_fft = n_fft
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.fs = 30  # 30 fps
    
    def _default_indices(self):
        """Return indices of 709 features to use for STFT."""
        # This depends on actual OpenFace CSV column order
        # Example mapping (ADJUST FOR YOUR CSV):
        indices = []
        
        # Head pose: columns 4-6 (pitch, roll, yaw)
        indices.extend([4, 5, 6])
        
        # Gaze: columns 7-10
        indices.extend([7, 8, 9, 10])
        
        # AUs: columns 15-31 (example: 17 AUs)
        indices.extend(range(15, 32))
        
        # Facial landmarks X,Y: columns 50-193 (example: 72 coordinates)
        indices.extend(range(50, 194))
        
        return indices
    
    def process_sample(self, frame_matrix):
        """
        Args:
            frame_matrix: (300, 709) numpy array
        
        Returns:
            stft_tensor: (n_features, n_freq_bins, n_time_windows)
        """
        selected = frame_matrix[:, self.feature_indices]  # (300, n_selected)
        
        stft_mags = []
        for col_idx in range(selected.shape[1]):
            signal = selected[:, col_idx]
            
            f, t, Zxx = scipy.signal.stft(
                signal,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.n_fft,
                window='hann'
            )
            mag = np.abs(Zxx)  # (n_freq_bins, n_time_windows)
            stft_mags.append(mag)
        
        # (n_features, n_freq_bins, n_time_windows)
        return np.array(stft_mags, dtype=np.float32)
    
    def process_dataset(self, X):
        """
        Args:
            X: (n_samples, 300, 709)
        
        Returns:
            X_spectral: (n_samples, n_features, n_freq_bins, n_time_windows)
        """
        spectral_list = []
        for i in range(X.shape[0]):
            spec = self.process_sample(X[i])
            spectral_list.append(spec)
        
        return np.array(spectral_list, dtype=np.float32)
```

**Integration into main pipeline**:

```python
# In main.py, add new mode:
if args.model == "spectral_cnn":
    logger.info("Extracting STFT features for spectral CNN...")
    
    spec_processor = SpectralPreprocessor(
        feature_indices=None,  # use defaults
        n_fft=64,
        nperseg=32,
        noverlap=16
    )
    
    X_train_spectral = spec_processor.process_dataset(X_train_normalized)
    X_test_spectral = spec_processor.process_dataset(X_test_normalized)
    
    # X_train_spectral shape: (n_train, n_features, 33, n_time_windows)
    X_train_input = X_train_spectral
    X_test_input = X_test_spectral
```

---

## 4. Training Strategy

### 4.1 Loss Function
Use **standard cross-entropy** initially:
```python
criterion = nn.CrossEntropyLoss()
```

If class imbalance persists, optionally add **weighted CE**:
```python
class_weights = compute_class_weights(y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### 4.2 Hyperparameters (Recommended Starting Values)

```yaml
epochs: 200
batch_size: 16
learning_rate: 1e-4
optimizer: Adam (beta1=0.9, beta2=0.999)
patience: 30  # early stopping
weight_decay: 1e-5
gradient_clip: 1.0  # prevent exploding gradients
```

**Why these values**:
- **Smaller learning rate (1e-4)**: Spectral features are more sensitive to learning rate
- **Larger batch size (16 vs 8)**: Spectral tensors are smaller in spatial dims; need batch averaging
- **Shorter patience (30 vs 50)**: Spectral features learn faster (fewer parameters than temporal CNN)

### 4.3 Training Loop (Pseudo-code)

```python
def train_spectral_cnn(model, train_loader, val_loader, epochs, device, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)  # (batch, n_features, 33, n_time_windows)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'spectral_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    
    model.load_state_dict(best_model_state)
    return model
```

---

## 5. Evaluation and Interpretability

### 5.1 Standard Metrics
Same as existing pipeline:
- **Accuracy**, **Macro-F1**, **Weighted-F1**
- **Per-class Precision/Recall/F1**
- **Confusion Matrix**

### 5.2 Spectral-Specific Analysis

**Visualization 1: Activation Maps**
```python
def visualize_spectral_activations(model, X_sample):
    """
    Show which frequency ranges activate for each engagement class.
    """
    # Extract intermediate activations (after spec_conv2)
    # Shape: (1, 128, 16, 2) for different freq/time regions
    # Visualize: heatmap of which freq bins are most active per class
    
    # For each class, show average activation map
    # X-axis: frequency (0 Hz to 15 Hz)
    # Y-axis: time (0 to 10 sec)
    # Color: activation magnitude
```

**Visualization 2: Spectrograms per Engagement Class**
```python
def plot_class_spectrograms(X_train, y_train, feature_idx=0):
    """
    Show average STFT spectrogram for each engagement class.
    
    For feature_idx (e.g., 0 = head yaw):
    - Plot 4 subplots (one per class)
    - Each subplot: 2D heatmap (frequency × time)
    - Interpretation: engaged students show power concentrated at low frequencies;
      disengaged show spread across high frequencies
    """
    for class_id in range(4):
        class_mask = y_train == class_id
        class_spectral = X_train[class_mask, feature_idx, :, :]  # (n_samples, freq, time)
        avg_spec = class_spectral.mean(axis=0)  # (freq, time)
        plt.subplot(2, 2, class_id + 1)
        plt.imshow(avg_spec, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"Class {class_id}: {ENGAGEMENT_LABELS[class_id]}")
        plt.xlabel("Time Windows")
        plt.ylabel("Frequency Bin (~0.5 Hz per bin)")
```

**Visualization 3: Feature Importance by Frequency**
```python
def freq_importance_per_class(model, X_test, y_test):
    """
    Measure: which frequency bands are most important for classification?
    
    Method: Compute gradient of logits w.r.t. spectral input, per frequency.
    
    Output: Bar chart
    - X-axis: frequency band (low/mid/high)
    - Y-axis: average gradient magnitude per engagement class
    - Interpretation: "Engaged detection relies on low-frequency stability"
    """
```

### 5.3 Comparison Report

Create a comparison table:

| Metric | Paper CNN | Temporal CNN | Spectral CNN |
|--------|-----------|-------------|-------------|
| Accuracy | ~71% | 76.8% | **Expected: 75-80%** |
| Macro-F1 | ~78% | 0.565 | **Expected: 0.58-0.68** |
| Class-0 Recall | ? | 0.37 | **Expected improvement** |
| Class-3 Recall | ? | 0.28 | **Expected improvement** |
| Inference time (ms/sample) | ? | ? | ~2-3 ms (faster than Transformer) |
| Model interpretability | Low | Low | **High** (frequency analysis) |

---

## 6. Implementation Checklist

### Phase 1: Feature Extraction (Week 1)
- [ ] Finalize feature indices mapping (extract from actual OpenFace CSV)
- [ ] Implement `SpectralPreprocessor.process_sample()` 
- [ ] Test STFT output shape: `(100, 33, 9)` expected
- [ ] Verify magnitude spectrograms look reasonable (visual inspection)
- [ ] Batch process train/test sets; save to disk for reuse

### Phase 2: Model Architecture (Week 1-2)
- [ ] Implement `SpectralConvNet` class
- [ ] Verify forward pass: `(batch, 100, 33, 9)` → `(batch, 4)` ✓
- [ ] Count parameters (should be ~200K-500K)
- [ ] Unit test with random input

### Phase 3: Integration (Week 2)
- [ ] Add `--model spectral_cnn` to argparse
- [ ] Add spectral preprocessing to main pipeline
- [ ] Test end-to-end: load data → preprocess → train → evaluate

### Phase 4: Training & Experiments (Week 2-3)
- [ ] Train baseline spectral CNN (cross-entropy loss)
- [ ] Train with weighted cross-entropy
- [ ] Compare vs temporal CNN
- [ ] Hyperparameter search: learning_rate, batch_size
- [ ] Early stopping with patience=30

### Phase 5: Analysis & Visualization (Week 3-4)
- [ ] Generate spectrograms per class
- [ ] Visualize frequency importance
- [ ] Compute frequency-domain statistics
- [ ] Write comparison report

### Phase 6: Thesis Documentation (Week 4)
- [ ] Write results section (implementation + findings)
- [ ] Create figures for thesis
- [ ] Analyze: does spectral approach improve minority classes?
- [ ] Interpret: what frequencies matter for engagement?

---

## 7. Expected Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| STFT output has NaN values | Clip signals to [-1e3, 1e3]; ensure normalization happens first |
| Spectral tensor memory is large | Reduce selected features from 100 to 50; use float32 not float64 |
| Model underfits (train/val loss both high) | Reduce dropout; increase model capacity (add channels) |
| Model overfits (train loss ↓, val loss ↑) | Increase dropout to 0.4-0.5; add L2 regularization |
| Macro-F1 still low (minority classes fail) | Add weighted loss or focal loss on top of spectral features |
| Training is slow | Pre-compute all spectral tensors; save to HDF5 for fast loading |

---

## 8. Code Skeleton (Complete Implementation Template)

Create a new file: `src/spectral_model.py`

```python
"""Spectral CNN model for engagement classification."""

import numpy as np
import scipy.signal
import torch
import torch.nn as nn


class SpectralPreprocessor:
    """Convert raw frame-feature sequences to spectral tensors via STFT."""
    
    def __init__(self, feature_indices=None, n_fft=64, nperseg=32, noverlap=16, fs=30):
        self.feature_indices = feature_indices or self._get_default_indices()
        self.n_fft = n_fft
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.fs = fs
    
    def _get_default_indices(self):
        """Return indices of 709 features to extract STFT from."""
        # TODO: Fill in based on your actual OpenFace CSV column mapping
        indices = []
        indices.extend([4, 5, 6])  # head pose
        indices.extend([7, 8, 9, 10])  # gaze
        indices.extend(range(15, 32))  # AUs
        # Add more as needed
        return indices
    
    def process_sample(self, frame_matrix):
        """
        Args: frame_matrix (300, 709)
        Returns: (n_selected_features, n_freq_bins, n_time_windows)
        """
        selected = frame_matrix[:, self.feature_indices]
        stft_mags = []
        
        for col in range(selected.shape[1]):
            signal = selected[:, col]
            f, t, Zxx = scipy.signal.stft(
                signal, fs=self.fs, nperseg=self.nperseg,
                noverlap=self.noverlap, nfft=self.n_fft, window='hann'
            )
            stft_mags.append(np.abs(Zxx))
        
        return np.array(stft_mags, dtype=np.float32)
    
    def process_dataset(self, X):
        """Args: X (n_samples, 300, 709)
        Returns: (n_samples, n_features, n_freq_bins, n_time_windows)
        """
        return np.array([self.process_sample(X[i]) for i in range(X.shape[0])])


class SpectralConvNet(nn.Module):
    """CNN over frequency-domain spectral features."""
    
    def __init__(self, n_input_features=100, n_freq_bins=33, num_classes=4):
        super().__init__()
        self.spec_conv1 = nn.Conv2d(n_input_features, 64, kernel_size=(5, 3), padding=(2, 1))
        self.spec_bn1 = nn.BatchNorm2d(64)
        self.spec_pool1 = nn.MaxPool2d((1, 2), stride=(1, 2))
        
        self.spec_conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.spec_bn2 = nn.BatchNorm2d(128)
        self.spec_pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self.spec_conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.spec_bn3 = nn.BatchNorm2d(128)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 128)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.spec_bn1(self.spec_conv1(x)))
        x = self.spec_pool1(x)
        
        x = torch.relu(self.spec_bn2(self.spec_conv2(x)))
        x = self.spec_pool2(x)
        
        x = torch.relu(self.spec_bn3(self.spec_conv3(x)))
        x = self.global_pool(x).view(x.size(0), -1)
        
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc_dropout(x)
        return self.fc2(x)
```

Then in `src/paper_repro_model.py`:
```python
def build_model(model_name, **kwargs):
    # ... existing code ...
    if model_name == "spectral_cnn":
        from src.spectral_model import SpectralConvNet
        return (
            SpectralConvNet(
                n_input_features=kwargs.get('n_input_features', 100),
                n_freq_bins=kwargs.get('n_freq_bins', 33),
                num_classes=4
            ),
            ModelSpec(name=model_name, input_kind="spectral")
        )
```

---

## 9. Success Criteria

The implementation is **successful** if:

✅ **Accuracy**: ≥ 74% (comparable to temporal CNN)
✅ **Macro-F1**: ≥ 0.55 (improves on temporal CNN's 0.565)
✅ **Class-0/3 Recall**: > 40% each (vs temporal's 37%/28%)
✅ **Interpretability**: Spectrograms show distinct patterns per class (frequency differences visible)
✅ **Speed**: Inference ≤ 5 ms/sample (reasonable for deployment)

**Bonus (thesis-level contribution)**:
- Frequency analysis reveals engagement signatures (e.g., "engaged = low-frequency head stability")
- Comparison visualization shows spectral CNN learns different patterns than temporal CNN
- Ablation study: remove certain frequency bands, measure performance drop

---

## 10. References & Inspiration

- **Spectral analysis in speech**: [Reference to voice emotion recognition]
- **EEG frequency bands**: [Theta, alpha, beta bands in attention studies]
- **Facial engagement papers**: Check if any use frequency-domain analysis
- **STFT implementations**: `scipy.signal.stft`, `librosa.stft`

---

## Summary

This specification provides a complete roadmap for implementing Temporal Engagement Signatures (TES) via spectral convolution. The approach is:
- **Novel**: Spectral analysis for facial engagement is underexplored
- **Interpretable**: Frequency bands reveal *how* engagement manifests
- **Feasible**: ~4-5 weeks implementation + evaluation
- **Publishable**: Unique angle; can generate interpretable visualizations

The implementation builds on your existing pipeline with minimal changes, and the spectral preprocessing can be integrated as an optional mode alongside temporal/rectangular CNN models.

