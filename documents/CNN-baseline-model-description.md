Below is a faithful implementation-oriented summary of the pipeline **starting from OpenFace features onward**, with every paper-stated design choice and hyperparameter preserved, and all missing details explicitly marked as unknown. Source: 

## 1. Input to the post-OpenFace pipeline

For each DAiSEE video:

* Duration: **10 seconds**
* Frame rate: **30 fps**
* Total frames per video: **300**
* OpenFace is run **per frame**
* Each frame produces a CSV row containing:

  * frame
  * face_id
  * timestamp
  * confidence
  * success
  * **709 facial feature values**
* The 709 OpenFace features are stated to cover:

  * facial landmark detection
  * head pose estimation
  * eye gaze estimation
  * facial expression / facial action units (AUs)
* The CSV is additionally modified to store:

  * file name
  * engagement label for each frame 

## 2. Label space

The task is **4-class engagement classification**:

* 0 = very low
* 1 = low
* 2 = high
* 3 = very high 

## 3. Data selection after OpenFace

The paper does **not** train on all 8,925 engagement videos after feature extraction. It performs a **two-stage data selection** process first.

### 3.1 Stage 1: video selection by person ID

The stated procedure is:

1. Find `id_people` from the video name using the **first five digits**
2. Search for unique `id_people`
3. Count number of videos for each unique `id_people`
4. Sort those counts in **ascending** order
5. Add cumulative counts until reaching threshold = **61**

   * 61 is the size of the minority class before balancing
6. Choose video names based on the selected unique `id_people` 

### 3.2 Stage 2: face selection within a frame

If OpenFace detects more than one face object in a frame:

* choose the object with the **highest confidence**
* example given:

  * face_id 0 confidence = 0.03
  * face_id 1 confidence = 0.98
  * keep face_id 1 

### 3.3 Dataset size after selection

Counts reported:

#### Before selection

* class 0: 61
* class 1: 455
* class 2: 4422
* class 3: 3987
* total: **8925**

#### After Stage 1 selection

* class 0: 61
* class 1: 63
* class 2: 70
* class 3: 76
* total: **270**

#### After Stage 2 selection

* class 0: 59
* class 1: 56
* class 2: 64
* class 3: 73
* total: **252** 

## 4. Per-video feature matrix shape before reduction

The paper implies the following:

* each video has **300 frames**
* each frame has **709 OpenFace features**
* therefore each selected video is represented as a **300 × 709** frame-feature matrix before dimensional reduction

This is also consistent with the CNN diagram text showing input as a feature matrix and with the statement that component = 300 was chosen so each video forms a **300 × 300** square feature matrix after reduction. 

## 5. Dimensionality reduction

Two separate experimental branches are run:

* **PCA-CNN**
* **SVD-CNN** 

### 5.1 Purpose

The paper states the 1×709 frame feature vector is too large, so dimensional reduction is used to obtain unique features that differentiate engagement levels. 

### 5.2 Algorithms used

* Principal Component Analysis (**PCA**)
* Singular Value Decomposition (**SVD**) 

### 5.3 Selection rule for number of components

The paper states:

* extracted components should cover a minimum of **80% explained variance**
* tested component counts include:

  * 2
  * 3
  * 10
  * 50
  * 100
  * 200
  * 300
* **300 components** were chosen because they gave the **highest explained variance** for both PCA and SVD
* choosing 300 also makes each video become a **300 × 300** square matrix
* reduction is therefore:

  * from **709 features/frame**
  * to **300 features/frame** 

### 5.4 Reported explained variance values

For PCA:

* 2 comps: 81.91727
* 3 comps: 96.99637
* 10 comps: 99.80054
* 50 comps: 99.99682
* 100 comps: 99.99963
* 200 comps: 99.99996
* 300 comps: 99.99998

For SVD:

* 2 comps: 72.52122
* 3 comps: 90.64553
* 10 comps: 99.77922
* 50 comps: 99.99676
* 100 comps: 99.99962
* 200 comps: 99.99996
* 300 comps: 99.99998 

### 5.5 Practical implication for implementation

Per selected video, after PCA or SVD:

* input becomes **300 frames × 300 reduced features**
* this is the matrix fed into the CNN as a 2D input matrix 

## 6. Normalization

After dimensional reduction:

* normalization method: **min-max normalization**
* output range: **[0, 1]**
* reason: reduced features have different value ranges and need normalization to avoid acting as noise during training 

## 7. Class balancing

After selection and preprocessing, the paper balances classes using **SMOTE**.

### 7.1 Method

* oversampling method: **SMOTE**
* purpose: synthesize minority-class samples to match the majority class 

### 7.2 Class counts before and after SMOTE

Before SMOTE:

* class 0: 59
* class 1: 56
* class 2: 64
* class 3: 73
* total: **252**

After SMOTE:

* class 0: 73
* class 1: 73
* class 2: 73
* class 3: 73
* total: **292** 

## 8. Train/test split

After balancing:

* train/test split = **80:20**
* training total = **233**
* testing total = **59**

Per class:

* class 0: train 58, test 15
* class 1: train 59, test 14
* class 2: train 58, test 15
* class 3: train 58, test 15 

## 9. CNN model architecture

The CNN is applied to the post-reduction feature matrix.

### 9.1 Input

The architecture figure indicates:

* input = **feature matrix**
* size shown in the figure is consistent with **300 × 300**
* dropout is shown before the classification head, with rate **0.25** 

### 9.2 Feature extraction backbone

The CNN feature extractor contains **4 convolution + pooling blocks**.

#### Block 1

* Conv2D
* number of feature maps = **32**
* kernel size = **5**
* activation = **ReLU**
* then MaxPooling
* pooling size = **2 × 2**

#### Block 2

* Conv2D
* number of feature maps = **64**
* kernel size = **5**
* activation = **ReLU**
* then MaxPooling
* pooling size = **2 × 2**

#### Block 3

* Conv2D
* number of feature maps = **128**
* kernel size = **5**
* activation = **ReLU**
* then MaxPooling
* pooling size = **2 × 2**

#### Block 4

* Conv2D
* number of feature maps = **256**
* kernel size = **5**
* activation = **ReLU**
* then MaxPooling
* pooling size = **2 × 2** 

### 9.3 Classification head

From the architecture figure:

* after conv/pooling blocks: **Dropout(0.25)**
* then **Flatten**
* then **Fully Connected / Dense**

  * units = **128**
  * activation = **ReLU**
* then another **Dropout(0.5)**
* then **Softmax classifier**

  * output units = **4** classes 

## 10. Training hyperparameters searched

The paper says parameter selection used **trial and error**.

The tested hyperparameter values were:

* **Optimizer**: Adam
* **Epochs**: 800, 1600
* **Batch size**: 32, 16, 8, 4, 2
* **Learning rate**: 1e-5, 1e-4 

## 11. Checkpointing and early stopping

The paper explicitly states both are used.

### 11.1 Checkpoint

* save model when loss decreases by a specified difference
* intended to retain the model with the lowest loss if later epochs stagnate or worsen

### 11.2 Early stopping

* stop when loss no longer shows significant decrease or model has converged
* patience parameter `p` is used
* **patience = half the number of epochs**

So:

* if epochs = 800, patience = **400**
* if epochs = 1600, patience = **800** 

## 12. Best-performing model settings

The paper reports best settings separately for PCA-CNN and SVD-CNN.

### 12.1 Best PCA-CNN model

Chosen best model: **Model 19**

* reduction: PCA
* optimizer: Adam
* epochs: **1600**
* learning rate: **1e-4**
* batch size: **4**
* average accuracy: **69.66**
* standard deviation: **3.34**
* minimum accuracy: **69.66**
* maximum accuracy: **72.88** 

Note: model 8 had higher max accuracy (74.58) but was not chosen because its std dev was larger (3.54 vs 3.34). 

### 12.2 Best SVD-CNN model

Chosen best model: **Model 38**

* reduction: SVD
* optimizer: Adam
* epochs: **1600**
* learning rate: **1e-4**
* batch size: **8**
* average accuracy: **71.02**
* standard deviation: **3.17**
* minimum accuracy: **67.8**
* maximum accuracy: **77.97** 

The paper says this model was stable across **10 iterations** per model. 

## 13. Precision / recall / F1 of the best models

### PCA-CNN Model 19

Per-class precision:

* class 0: 0.62
* class 1: 0.73
* class 2: 0.92
* class 3: 0.73
* average: **0.75**

Per-class recall:

* class 0: 0.87
* class 1: 0.57
* class 2: 0.73
* class 3: 0.73
* average: **0.73**

Per-class F1:

* class 0: 0.72
* class 1: 0.64
* class 2: 0.81
* class 3: 0.73
* average: **0.73** 

### SVD-CNN Model 38

Per-class precision:

* class 0: 0.85
* class 1: 0.83
* class 2: 0.81
* class 3: 0.67
* average: **0.79**

Per-class recall:

* class 0: 0.73
* class 1: 0.71
* class 2: 0.87
* class 3: 0.80
* average: **0.78**

Per-class F1:

* class 0: 0.79
* class 1: 0.77
* class 2: 0.84
* class 3: 0.73
* average: **0.78** 

## 14. Exact reproduction notes for a coding agent

To reproduce the paper as closely as possible:

1. Run OpenFace framewise on each selected DAiSEE video.
2. For each frame, retain the row containing metadata plus **709 facial features**.
3. Attach video filename and engagement label to frame records.
4. Use the person-ID-based video selection procedure described above.
5. If multiple faces are present in a frame, keep the one with **maximum confidence**.
6. Build one per-video matrix of shape **300 × 709**.
7. Run either PCA or SVD to reduce the feature dimension from **709 to 300**.
8. Ensure final per-video matrix becomes **300 × 300**.
9. Apply **min-max normalization to [0,1]**.
10. After preprocessing, apply **SMOTE** to class counts until every class has **73 samples**.
11. Split balanced data **80/20** into train/test.
12. Train CNN with:

    * 4 Conv blocks
    * feature maps: 32, 64, 128, 256
    * kernel size 5
    * ReLU
    * max-pool 2×2 after each conv
    * dropout 0.25
    * flatten
    * dense 128 ReLU
    * dropout 0.5
    * softmax output of size 4
13. Use Adam optimizer.
14. Search over:

    * epochs: 800 or 1600
    * learning rate: 1e-5 or 1e-4
    * batch size: 32, 16, 8, 4, 2
15. Use checkpointing and early stopping with patience = **epochs / 2**.
16. Best reported configuration for the final paper result:

    * **SVD + CNN**
    * epochs = **1600**
    * lr = **1e-4**
    * batch size = **8**
    * optimizer = **Adam**
    * max accuracy = **77.97%** 

## 15. Critical missing details not specified in the paper

These are **not stated** in the paper and must be treated as unknown unless you inspect authors’ code:

* exact OpenFace command line / version
* exact list and ordering of the 709 features
* whether failed OpenFace frames were removed or retained
* how missing frames or invalid detections were handled
* whether PCA/SVD was fit:

  * globally across all frames from all videos
  * per video
  * train-only then applied to test
* exact tensor layout given to CNN:

  * whether treated as grayscale image-like tensor `(300,300,1)`
  * or another layout
* conv padding type (`same` vs `valid`)
* conv stride
* pooling stride
* dense bias settings
* weight initialization
* loss function name

  * likely categorical cross-entropy, but **not explicitly stated**
* whether labels were one-hot encoded
* whether train/test split was stratified
* exact random seeds
* exact checkpoint threshold for “loss decreases by a specified difference”
* exact early stopping monitor metric
* whether SMOTE was applied before or after train/test split in implementation

  * the paper narrative suggests preprocessing/balancing before split, but this is methodologically risky; still, faithful reproduction should follow the paper wording unless code shows otherwise
* whether the reported 10 iterations use different random splits or only different initializations 

## 16. Recommended faithful implementation policy when paper is silent

For a coding agent trying to reproduce the paper, use this rule:

* follow all paper-stated values exactly
* for missing details, mark them as implementation assumptions and keep them conservative

Suggested assumptions for a reproduction run:

* input tensor shape: `(300, 300, 1)`
* Conv2D padding: `same`
* stride: 1
* MaxPool stride: 2
* loss: categorical cross-entropy
* output activation: softmax
* early stopping monitor: validation loss
* checkpoint monitor: validation loss
* save_best_only: true
* SMOTE applied exactly where the paper places it in the pipeline, i.e. **before the final 80/20 split**, if strict faithfulness is prioritized over best practice

That last point is an inference from the paper’s wording, not an explicit statement. 

If you want, I can turn this into a **strict implementation spec file** formatted for Copilot/Claude Code, with sections like `INPUT`, `PREPROCESSING`, `MODEL`, `TRAINING`, `UNKNOWNS`, and `ASSUMPTIONS`.
