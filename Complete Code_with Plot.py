# -*- coding: utf-8 -*-
"""Ulite_Updated_Final_spc_perc.ipynb
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install spectral
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# ================================
# ULITE MODEL(Memory-Safe, Same Architecture/Hyperparams)
# ================================
# !pip install spectral

import os, time, zipfile, gc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
import spectral

# --------- Config ---------
dataset = 'Ho'                 # Salinas
windowSize = 25                # keep as requested
K = 30                         # PCA components
train_ratio = 0.05             # 10% per class for training
batch_size = 256               # keep same as your code
epochs = 100                   # keep same
model_name = "IP_PCA_30_ULITE_perc_5_Optimized"
results_folder = "ulite_results_Ho_per5"
os.makedirs(results_folder, exist_ok=True)

# Optional mixed precision to save memory
mixed_precision.set_global_policy("mixed_float16")
tf.keras.backend.clear_session()
gc.collect()

# --------- Data Loading ---------
def loadData(name):
    # Adjust base path if your files are elsewhere
    base = "/content/drive/MyDrive/Colab Notebooks/dataset/"  # e.g., "/content/drive/MyDrive/Colab Notebooks/dataset/"
    if name == 'IP':
        data = sio.loadmat(os.path.join(base, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(base, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(base, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(base, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'Ho':
        data = sio.loadmat(os.path.join(base, 'Houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(base, 'Houston_gt.mat'))['houston_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(base, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(base, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'Bo':
        data = sio.loadmat(os.path.join(base, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(base, 'Botswana_gt.mat'))['Botswana_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(base, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(base, 'KSC_gt.mat'))['KSC_gt']
    else:
        raise ValueError("Only 'Ho' (Houston) implemented in this script.")
    return data, labels

def applyPCA(X, numComponents):
    Xr = X.reshape(-1, X.shape[2]).astype(np.float32)
    pca = PCA(n_components=numComponents, whiten=True)
    Xp = pca.fit_transform(Xr)
    return Xp.reshape(X.shape[0], X.shape[1], numComponents), pca

def padWithZeros(X, margin=0):
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

# --------- Memory-Safe Patch Generator ---------
class PatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, coords, labels, full_cube, patch_size=25,
                 batch_size=256, shuffle=True, n_classes=15):
        self.coords = coords
        self.labels = labels
        self.full_cube = full_cube
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.half = patch_size // 2
        self.indices = np.arange(len(labels))
        self.padded = padWithZeros(full_cube, self.half)  # pad once
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # preallocate for speed
        batch_X = np.empty((len(idxs), self.patch_size, self.patch_size, self.full_cube.shape[2], 1), dtype=np.float32)
        batch_y = np.empty((len(idxs),), dtype=np.int32)
        for m, k in enumerate(idxs):
            r, c = self.coords[k]
            patch = self.padded[r:r+self.patch_size, c:c+self.patch_size, :]
            batch_X[m, ..., 0] = patch
            batch_y[m] = self.labels[k]
        # NOTE: to_categorical returns float; mixed precision will cast as needed
        batch_y = to_categorical(batch_y, num_classes=self.n_classes)
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --------- Per-Class Split on Coordinates ---------
def splitTrainTestCoords(coords, labels, train_ratio=0.1, random_state=42):
    np.random.seed(random_state)
    train_idx, test_idx = [], []
    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    return coords[train_idx], coords[test_idx], labels[train_idx], labels[test_idx]

# ---------------- ULite-R2HCN blocks (simplified/robust) ----------------
from tensorflow.keras import layers
def R2SpectralBlock(x, groups=4):
    """
    Lightweight spectral block: grouped 1x1 spectral mixing followed by small conv.
    groups: number of groups for group conv to reduce redundancy.
    """
    # x shape: (H, W, Bands, C) or (S,S,K,channels) - we operate on last axis channels
    # Use pointwise grouped conv across spectral axis: implement as Conv2D with kernel (1,1)
    # but we want to mix spectral bands — since input patch shape is (S,S,K,1),
    # we'll squeeze channels dimension and use Conv3D along spectral dimension if needed.
    # Simpler: use a 1x1x1 Conv3D grouped via channel split to simulate grouped spectral mixing.
    in_channels = x.shape[-1]
    # pointwise conv (1x1) along spatial dims and 1x1 spectral mixing via Conv3D with kernel (1,1,1)
    # We'll simulate grouping by splitting channels and applying small Dense-like mixing
    # Implementation chosen for compactness:
    y = layers.Conv3D(filters=max(4, in_channels), kernel_size=(1,1,1), padding='same', activation='relu')(x)
    y = layers.BatchNormalization()(y)
    # 1x1 spectral projection (reduce channels)
    y = layers.Conv3D(filters=max(4, in_channels//2 + 1), kernel_size=(1,1,1), padding='same', activation='relu')(y)
    y = layers.BatchNormalization()(y)
    return y

def R2SpatialBlock(x):
    """
    Lightweight spatial block: depthwise-separable-like 3D conv with small kernels to capture local spatial context.
    """
    # Apply small 3D conv followed by a separable 2D conv
    y = layers.Conv3D(filters= max(8, int(x.shape[-1]) ), kernel_size=(3,3,3), padding='same', activation='relu')(x)
    y = layers.BatchNormalization()(y)
    # Squeeze spectral dimension by a 1x1 conv then apply 2D separable conv
    # reshape from (S,S,K,C) -> (S,S,K*C,1) not necessary; we apply Conv2D on last two dims after reshape
    shape = y.shape
    # collapse spectral and channel dims for Conv2D processing
    # Use Keras operations for getting shape components
    bsize = tf.keras.ops.shape(y)[0]
    h = tf.keras.ops.shape(y)[1]
    w = tf.keras.ops.shape(y)[2]
    d = tf.keras.ops.shape(y)[3]
    ch = tf.keras.ops.shape(y)[4]
    # merge spectral & channel dims
    y_resh = layers.Reshape((h, w, d*ch))(y)
    y_resh = layers.SeparableConv2D(filters=max(8, ch*2), kernel_size=(3,3), padding='same', activation='relu')(y_resh)
    y_resh = layers.BatchNormalization()(y_resh)
    # project back to a compact 3D-like shape by adding a spectral pseudo-dim (1)
    y_out = layers.Reshape((h, w, 1, int(y_resh.shape[-1]//1)))(y_resh)
    return y_out

# ---------- Build ULite-R2HCN model ----------
def build_ulite_r2hcn(windowSize, K, num_classes):
    # Input shape: (S,S,K,1)
    inp = layers.Input(shape=(windowSize, windowSize, K, 1), dtype='float32')
    # initial spectral-reduction conv (light)
    x = layers.Conv3D(filters=8, kernel_size=(3,3,7), padding='valid', activation='relu')(inp)   # small spectral kernel
    x = layers.Conv3D(filters=16, kernel_size=(3,3,5), padding='valid', activation='relu')(x)
    # apply R2SpectralBlock
    x = R2SpectralBlock(x)
    # R2SpatialBlock
    x = R2SpatialBlock(x)
    # further light convs
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(layers.Reshape((x.shape[1], x.shape[2], x.shape[3]*x.shape[4]))(x))
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    # small classifier head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # cast to float32 for numeric stability
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
'''

# ---------------- Pipeline ----------------
print("Loading data...")
X_full, y_full = loadData(dataset)
n_classes = int(y_full.max())
print(f"Dataset {dataset}: H={X_full.shape[0]}, W={X_full.shape[1]}, Bands={X_full.shape[2]}, Classes={n_classes}")
total_labeled = int(np.sum(y_full > 0))
print("Total labeled pixels in full map:", total_labeled)

print("Applying PCA ...")
X_pca, pca = applyPCA(X_full, numComponents=K)

# collect coords & labels for ALL labeled pixels (include edges)
coords, labels = [], []
H_img, W_img = X_pca.shape[0], X_pca.shape[1]
for r in range(0, H_img):
    for c in range(0, W_img):
        lab = y_full[r, c]
        if lab > 0:
            coords.append((r, c))          # original coordinates (centers)
            labels.append(lab - 1)         # zero-based labels
coords = np.array(coords, dtype=np.int32)
labels = np.array(labels, dtype=np.int32)
print("Collected coords (should equal total labeled):", len(labels))'''


###############################################################################

# =========================
# Pipeline
# =========================
# 1) Load + PCA
X_full, y_full = loadData(dataset)
n_classes = int(y_full.max())  # 16 for Salinas
X_pca, pca = applyPCA(X_full, numComponents=K)

# 2) Build coordinate list (only labeled pixels with border safety)
margin = windowSize // 2
coords, labels = [], []
for r in range(margin, X_pca.shape[0]-margin):
    for c in range(margin, X_pca.shape[1]-margin):
        lab = y_full[r, c]
        if lab > 0:
            coords.append((r, c))
            labels.append(lab - 1)  # zero-based
coords = np.array(coords, dtype=np.int32)
labels = np.array(labels, dtype=np.int32)

# 3) Train/Test split (10% per class)
train_coords, test_coords, ytrain_idx, ytest_idx = splitTrainTestCoords(coords, labels, train_ratio)
print("Train samples:", len(ytrain_idx), "Test samples:", len(ytest_idx))
# 4) Generators
train_gen = PatchGenerator(train_coords, ytrain_idx, X_pca, patch_size=windowSize,
                           batch_size=batch_size, shuffle=True, n_classes=n_classes)
test_gen  = PatchGenerator(test_coords,  ytest_idx,  X_pca, patch_size=windowSize,
                           batch_size=batch_size, shuffle=False, n_classes=n_classes)

# 5) Model
model = build_model(windowSize, K, n_classes)

# Save model summary
with open(os.path.join(results_folder, f"{model_name}_summary.txt"), "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# 6) Train
tic = time.perf_counter()
history = model.fit(train_gen, validation_data=test_gen, epochs=epochs, verbose=2)
toc = time.perf_counter()
train_time = toc - tic

# 7) Plots: accuracy & loss
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Value"); plt.legend()
plt.title("Training vs Validation")
plt.savefig(os.path.join(results_folder, f"{model_name}_training.png"), dpi=150)
plt.close()

# 8) Evaluation on Test Set
tic1 = time.perf_counter()
y_pred_prob = model.predict(test_gen, verbose=0)
toc1 = time.perf_counter()
test_time = toc1 - tic1

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = ytest_idx  # already zero-based labels

# Metrics
classification = classification_report(y_true, y_pred, digits=4)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
oa = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
each_acc = np.nan_to_num(np.diag(cm) / cm.sum(axis=1, keepdims=False))
aa = np.mean(each_acc)
kappa = cohen_kappa_score(y_true, y_pred)

# Confusion matrix figure
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap='viridis')
plt.title('Confusion Matrix'); plt.colorbar()
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, f"{model_name}_confusion_matrix.png"), dpi=150)
plt.close()

# 9) Save Results Text (all metrics + model desc)
with open(os.path.join(results_folder, f"{model_name}_results.txt"), "w") as f:
    f.write(f"Model: 3D→2D CNN (Conv3D(8,3x3x7)->Conv3D(16,3x3x5)->Conv3D(32,3x3x3)"
            f"->Reshape->Conv2D(24,3x3)->Conv2D(96,3x3)->MaxPool2D(2x2)->Conv2D(128,3x3)"
            f"->GAP->Dense(128)->Dropout(0.4)->Dense(64)->Dropout(0.4)->Dense({n_classes}))\n")
    f.write(f"Optimizer: Adam(1e-3), Batch Size: {batch_size}, Epochs: {epochs}\n")
    f.write(f"Window Size: {windowSize}x{windowSize}, PCA Components: {K}\n")
    f.write(f"Training ratio per class: {int(train_ratio*100)}%\n\n")

    f.write(f"Training time: {train_time:.2f} s\n")
    f.write(f"Testing time: {test_time:.2f} s\n\n")

    f.write(f"Overall Accuracy: {oa*100:.2f}%\n")
    f.write(f"Average Accuracy: {aa*100:.2f}%\n")
    f.write(f"Kappa: {kappa*100:.2f}%\n")
    f.write(f"Precision (weighted): {precision*100:.2f}%\n")
    f.write(f"Recall (weighted): {recall*100:.2f}%\n")
    f.write(f"F1-score (weighted): {f1*100:.2f}%\n\n")

    f.write("Classwise Accuracy (%):\n")
    f.write(", ".join([f"{x*100:.2f}" for x in each_acc]) + "\n\n")

    f.write("Classification Report:\n")
    f.write(classification + "\n\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm) + "\n")

# 10) Full Map Prediction (batched, memory-safe)
print("Predicting full map (batched)...")
PATCH = windowSize
pad = PATCH // 2
Xp = padWithZeros(X_pca, pad)
H, W = y_full.shape
outputs = np.zeros((H, W), dtype=np.int32)

# Collect coords of labeled pixels
coords_all = [(r, c) for r in range(pad, H-pad) for c in range(pad, W-pad) if y_full[r, c] > 0]
B = 2048  # batch for full-map inference
for start in range(0, len(coords_all), B):
    batch_coords = coords_all[start:start+B]
    batch = np.empty((len(batch_coords), PATCH, PATCH, K, 1), dtype=np.float32)
    for i, (r, c) in enumerate(batch_coords):
        patch = Xp[r-pad:r+pad+1, c-pad:c+pad+1, :]
        batch[i, ..., 0] = patch
    preds = np.argmax(model.predict(batch, verbose=0), axis=1)
    for (r, c), p in zip(batch_coords, preds):
        outputs[r, c] = p + 1  # back to 1..C

# Save classified and ground-truth maps
spectral.save_rgb(os.path.join(results_folder, f"{model_name}_classified_map.jpg"),
                  outputs.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(os.path.join(results_folder, f"{model_name}_groundtruth.jpg"),
                  y_full, colors=spectral.spy_colors)

# 11) Zip everything
zip_path = f"{model_name}_outputs.zip"
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for fn in os.listdir(results_folder):
        zipf.write(os.path.join(results_folder, fn), arcname=fn)
np.save(os.path.join(results_folder, "ulite_history.npy"), history.history)
print(f"✅ Done. All outputs saved in: {zip_path}")

# ==========PROPOSED LhSSN======================
# Salinas HSI (Memory-Safe, Same Architecture/Hyperparams)
# ================================
# !pip install spectral

import os, time, zipfile, gc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
import spectral

# --------- Config ---------
dataset = 'Ho'                 # Salinas
windowSize = 25                # keep as requested
K = 15                         # PCA components
train_ratio = 0.05            # 10% per class for training
batch_size = 256               # keep same as your code
epochs = 100                   # keep same
model_name = "Ho_PCA_15_Light_perc_5_Optimized"
results_folder = "ulite_results_Ho_per5"
os.makedirs(results_folder, exist_ok=True)

# Optional mixed precision to save memory
mixed_precision.set_global_policy("mixed_float16")
tf.keras.backend.clear_session()
gc.collect()

# --------- Data Loading ---------
def loadData(name):
    # Adjust base path if your files are elsewhere
    base = "/content/drive/MyDrive/Colab Notebooks/dataset/"  # e.g., "/content/drive/MyDrive/Colab Notebooks/dataset/"
    if name == 'Ho':
        data   = sio.loadmat(os.path.join(base, 'Houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(base, 'Houston_gt.mat'))['houston_gt']
    else:
        raise ValueError("Only 'Ho' (Houston) implemented in this script.")
    return data, labels

def applyPCA(X, numComponents):
    Xr = X.reshape(-1, X.shape[2]).astype(np.float32)
    pca = PCA(n_components=numComponents, whiten=True)
    Xp = pca.fit_transform(Xr)
    return Xp.reshape(X.shape[0], X.shape[1], numComponents), pca

def padWithZeros(X, margin=0):
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

# --------- Memory-Safe Patch Generator ---------
class PatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, coords, labels, full_cube, patch_size=25,
                 batch_size=256, shuffle=True, n_classes=20):
        self.coords = coords
        self.labels = labels
        self.full_cube = full_cube
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.half = patch_size // 2
        self.indices = np.arange(len(labels))
        self.padded = padWithZeros(full_cube, self.half)  # pad once
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # preallocate for speed
        batch_X = np.empty((len(idxs), self.patch_size, self.patch_size, self.full_cube.shape[2], 1), dtype=np.float32)
        batch_y = np.empty((len(idxs),), dtype=np.int32)
        for m, k in enumerate(idxs):
            r, c = self.coords[k]
            patch = self.padded[r:r+self.patch_size, c:c+self.patch_size, :]
            batch_X[m, ..., 0] = patch
            batch_y[m] = self.labels[k]
        # NOTE: to_categorical returns float; mixed precision will cast as needed
        batch_y = to_categorical(batch_y, num_classes=self.n_classes)
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --------- Per-Class Split on Coordinates ---------
def splitTrainTestCoords(coords, labels, train_ratio=0.1, seed=42):
    np.random.seed(seed)
    train_idx, test_idx = [], []
    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    return coords[train_idx], coords[test_idx], labels[train_idx], labels[test_idx]

# --------- Build Model (EXACT architecture you used) ---------
def build_model(S, L, num_classes):
    inp = tf.keras.layers.Input((S, S, L, 1))
    x = tf.keras.layers.Conv3D(filters=8,  kernel_size=(3,3,7), activation='relu')(inp)
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,5), activation='relu')(x)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(x)
    # reshape to 2D convs
    conv3d_shape = x.shape  # (None, H, W, D, C)
    x = tf.keras.layers.Reshape((conv3d_shape[1], conv3d_shape[2],
                                 conv3d_shape[3]*conv3d_shape[4]))(x)
    x = tf.keras.layers.Conv2D(filters=24,  kernel_size=(3,3), activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=96,  kernel_size=(3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    # Adam 0.001 as in your code
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# =========================
# Pipeline
# =========================
# 1) Load + PCA
X_full, y_full = loadData(dataset)
n_classes = int(y_full.max())  # 16 for Salinas
X_pca, pca = applyPCA(X_full, numComponents=K)

# 2) Build coordinate list (only labeled pixels with border safety)
margin = windowSize // 2
coords, labels = [], []
for r in range(margin, X_pca.shape[0]-margin):
    for c in range(margin, X_pca.shape[1]-margin):
        lab = y_full[r, c]
        if lab > 0:
            coords.append((r, c))
            labels.append(lab - 1)  # zero-based
coords = np.array(coords, dtype=np.int32)
labels = np.array(labels, dtype=np.int32)

# 3) Train/Test split (10% per class)
train_coords, test_coords, ytrain_idx, ytest_idx = splitTrainTestCoords(coords, labels, train_ratio=train_ratio)

# 4) Generators
train_gen = PatchGenerator(train_coords, ytrain_idx, X_pca, patch_size=windowSize,
                           batch_size=batch_size, shuffle=True, n_classes=n_classes)
test_gen  = PatchGenerator(test_coords,  ytest_idx,  X_pca, patch_size=windowSize,
                           batch_size=batch_size, shuffle=False, n_classes=n_classes)

# 5) Model
model = build_model(windowSize, K, n_classes)

# Save model summary
with open(os.path.join(results_folder, f"{model_name}_summary.txt"), "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# 6) Train
tic = time.perf_counter()
history = model.fit(train_gen, validation_data=test_gen, epochs=epochs, verbose=2)
toc = time.perf_counter()
train_time = toc - tic

# 7) Plots: accuracy & loss
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Value"); plt.legend()
plt.title("Training vs Validation")
plt.savefig(os.path.join(results_folder, f"{model_name}_training.png"), dpi=150)
plt.close()

# 8) Evaluation on Test Set
tic1 = time.perf_counter()
y_pred_prob = model.predict(test_gen, verbose=0)
toc1 = time.perf_counter()
test_time = toc1 - tic1

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = ytest_idx  # already zero-based labels

# Metrics
classification = classification_report(y_true, y_pred, digits=4)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
oa = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
each_acc = np.nan_to_num(np.diag(cm) / cm.sum(axis=1, keepdims=False))
aa = np.mean(each_acc)
kappa = cohen_kappa_score(y_true, y_pred)

# Confusion matrix figure
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap='viridis')
plt.title('Confusion Matrix'); plt.colorbar()
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, f"{model_name}_confusion_matrix.png"), dpi=150)
plt.close()

# 9) Save Results Text (all metrics + model desc)
with open(os.path.join(results_folder, f"{model_name}_results.txt"), "w") as f:
    f.write(f"Model: 3D→2D CNN (Conv3D(8,3x3x7)->Conv3D(16,3x3x5)->Conv3D(32,3x3x3)"
            f"->Reshape->Conv2D(24,3x3)->Conv2D(96,3x3)->MaxPool2D(2x2)->Conv2D(128,3x3)"
            f"->GAP->Dense(128)->Dropout(0.4)->Dense(64)->Dropout(0.4)->Dense({n_classes}))\n")
    f.write(f"Optimizer: Adam(1e-3), Batch Size: {batch_size}, Epochs: {epochs}\n")
    f.write(f"Window Size: {windowSize}x{windowSize}, PCA Components: {K}\n")
    f.write(f"Training ratio per class: {int(train_ratio*100)}%\n\n")

    f.write(f"Training time: {train_time:.2f} s\n")
    f.write(f"Testing time: {test_time:.2f} s\n\n")

    f.write(f"Overall Accuracy: {oa*100:.2f}%\n")
    f.write(f"Average Accuracy: {aa*100:.2f}%\n")
    f.write(f"Kappa: {kappa*100:.2f}%\n")
    f.write(f"Precision (weighted): {precision*100:.2f}%\n")
    f.write(f"Recall (weighted): {recall*100:.2f}%\n")
    f.write(f"F1-score (weighted): {f1*100:.2f}%\n\n")

    f.write("Classwise Accuracy (%):\n")
    f.write(", ".join([f"{x*100:.2f}" for x in each_acc]) + "\n\n")

    f.write("Classification Report:\n")
    f.write(classification + "\n\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm) + "\n")

# 10) Full Map Prediction (batched, memory-safe)
print("Predicting full map (batched)...")
PATCH = windowSize
pad = PATCH // 2
Xp = padWithZeros(X_pca, pad)
H, W = y_full.shape
outputs = np.zeros((H, W), dtype=np.int32)

# Collect coords of labeled pixels
coords_all = [(r, c) for r in range(pad, H-pad) for c in range(pad, W-pad) if y_full[r, c] > 0]
B = 2048  # batch for full-map inference
for start in range(0, len(coords_all), B):
    batch_coords = coords_all[start:start+B]
    batch = np.empty((len(batch_coords), PATCH, PATCH, K, 1), dtype=np.float32)
    for i, (r, c) in enumerate(batch_coords):
        patch = Xp[r-pad:r+pad+1, c-pad:c+pad+1, :]
        batch[i, ..., 0] = patch
    preds = np.argmax(model.predict(batch, verbose=0), axis=1)
    for (r, c), p in zip(batch_coords, preds):
        outputs[r, c] = p + 1  # back to 1..C

# Save classified and ground-truth maps
spectral.save_rgb(os.path.join(results_folder, f"{model_name}_classified_map.jpg"),
                  outputs.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(os.path.join(results_folder, f"{model_name}_groundtruth.jpg"),
                  y_full, colors=spectral.spy_colors)

# 11) Zip everything
zip_path = f"{model_name}_outputs.zip"
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for fn in os.listdir(results_folder):
        zipf.write(os.path.join(results_folder, fn), arcname=fn)
np.save(os.path.join(results_folder, "lhSSN_history.npy"), history.history)
print(f"✅ Done. All outputs saved in: {zip_path}")

# ====C-VIT CODE========================================================
# INSTALL DEPENDENCIES
# ============================================================
!pip install -q spectral einops

# ============================================================
# IMPORTS
# ============================================================
import os, time, zipfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    classification_report, cohen_kappa_score,
    precision_recall_fscore_support
)
from tensorflow.keras.utils import to_categorical
from einops.layers.tensorflow import Rearrange
import spectral
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Zeros

# ============================================================
# CONFIGURATION
# ============================================================
dataset = 'Ho'                     # IP, SA, PU, KSC, Bo, Ho
windowSize = 25
PCA_BANDS = 15
epochs = 100
batch_size = 256
train_ratio = 0.05                 # percentage-wise split
samples_per_class = None           # set integer for count-wise split

model_name = "Ho_PCA15_ViT_5_spc"
results_folder = "ulite_results_Ho_per5"
os.makedirs(results_folder, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
def loadData(name):
    base = "/content/drive/MyDrive/Colab Notebooks/dataset/"
    if name == 'IP':
        data = sio.loadmat(base+'Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(base+'Indian_pines_gt.mat')['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(base, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(base, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'Ho':
        data = sio.loadmat(os.path.join(base, 'Houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(base, 'Houston_gt.mat'))['houston_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(base, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(base, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'Bo':
        data = sio.loadmat(os.path.join(base, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(base, 'Botswana_gt.mat'))['Botswana_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(base, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(base, 'KSC_gt.mat'))['KSC_gt']
    else:
        raise ValueError("Dataset not configured")
    return data, labels

# ============================================================
# PREPROCESSING
# ============================================================
def applyPCA(X, n_components):
    Xr = X.reshape(-1, X.shape[2])
    pca = PCA(n_components=n_components, whiten=True)
    Xp = pca.fit_transform(Xr)
    return Xp.reshape(X.shape[0], X.shape[1], n_components)

def padWithZeros(X, margin):
    return np.pad(X, ((margin, margin),(margin, margin),(0,0)), mode='constant')

def createImageCubes(X, y, windowSize):
    margin = windowSize // 2
    Xp = padWithZeros(X, margin)
    patches, labels = [], []
    for r in range(margin, Xp.shape[0]-margin):
        for c in range(margin, Xp.shape[1]-margin):
            if y[r-margin, c-margin] > 0:
                patches.append(Xp[r-margin:r+margin+1, c-margin:c+margin+1])
                labels.append(y[r-margin, c-margin]-1)
    return np.array(patches), np.array(labels)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
def splitTrainTestSet(X, y, train_ratio=None, samples_per_class=None):
    np.random.seed(42)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y==c)[0]
        np.random.shuffle(idx)
        n = min(samples_per_class, len(idx)-1) if samples_per_class else max(1,int(len(idx)*train_ratio))
        tr.extend(idx[:n]); te.extend(idx[n:])
    return X[tr], X[te], y[tr], y[te]

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
X, y = loadData(dataset)
X = applyPCA(X, PCA_BANDS)
X, y = createImageCubes(X, y, windowSize)

Xtr, Xte, ytr, yte = splitTrainTestSet(
    X, y, train_ratio=train_ratio, samples_per_class=samples_per_class
)

ytr_cat = to_categorical(ytr)
yte_cat = to_categorical(yte)

# Custom layer to add and repeat the class token
class AddClassTokenLayer(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.cls_token = self.add_weight(
            shape=(1, 1, dim),
            initializer=Zeros(),
            trainable=True,
            name='cls_token_weight'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        repeated_cls_token = tf.repeat(self.cls_token, batch_size, axis=0)
        return tf.concat([repeated_cls_token, inputs], axis=1)

# ============================================================
# VISION TRANSFORMER MODEL
# ============================================================
def build_vit(image_size=25, patch_size=5, channels=30,
              num_classes=16, dim=64, depth=6, heads=4, mlp_dim=128):

    num_patches = (image_size // patch_size) ** 2

    inputs = tf.keras.Input(shape=(image_size,image_size,channels))

    x = Rearrange(
        'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
        p1=patch_size, p2=patch_size
    )(inputs)

    x = tf.keras.layers.Dense(dim)(x)

    # Use the custom layer for the class token
    x = AddClassTokenLayer(dim=dim)(x)

    pos_embed = tf.keras.layers.Embedding(num_patches+1, dim)
    x = x + pos_embed(tf.range(num_patches+1))

    for _ in range(depth):
        x1 = tf.keras.layers.LayerNormalization()(x)
        attn = tf.keras.layers.MultiHeadAttention(heads, dim)(x1,x1)
        x = x + attn
        x2 = tf.keras.layers.LayerNormalization()(x)
        mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation='gelu'),
            tf.keras.layers.Dense(dim)
        ])(x2)
        x = x + mlp

    x = tf.keras.layers.LayerNormalization()(x)
    x = x[:,0]
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_vit(
    image_size=windowSize,
    patch_size=5,
    channels=PCA_BANDS,
    num_classes=ytr_cat.shape[1]
)

model.summary()

# ============================================================
# TRAINING
# ============================================================
tic = time.perf_counter()
history = model.fit(
    Xtr, ytr_cat,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(Xte, yte_cat),
    verbose=2
)
train_time = time.perf_counter() - tic

# ============================================================
# TRAINING CURVES
# ============================================================
'''plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs"); plt.ylabel("Value"); plt.legend()
plt.savefig(os.path.join(results_folder, model_name+"_training.png"))
plt.close()'''
#############################################
# 7) Plots: accuracy & loss
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Value"); plt.legend()
plt.title("Training vs Validation")
plt.savefig(os.path.join(results_folder, f"{model_name}_training.png"), dpi=150)
plt.close()
# ============================================================
# EVALUATION
# ============================================================
tic = time.perf_counter()
pred = np.argmax(model.predict(Xte), axis=1)
test_time = time.perf_counter() - tic

oa = accuracy_score(yte, pred)
cm = confusion_matrix(yte, pred)
each_acc = np.nan_to_num(np.diag(cm)/cm.sum(axis=1))
aa = np.mean(each_acc)
kappa = cohen_kappa_score(yte, pred)

report = classification_report(yte, pred, digits=4)

# ============================================================
# SAVE RESULTS
# ============================================================
with open(os.path.join(results_folder, model_name+"_results.txt"), "w") as f:
    f.write(f"Training time: {train_time:.2f}s\n")
    f.write(f"Testing time: {test_time:.2f}s\n")
    f.write(f"OA: {oa*100:.2f}%\n")
    f.write(f"AA: {aa*100:.2f}%\n")
    f.write(f"Kappa: {kappa*100:.2f}%\n\n")
    f.write(report)

# ============================================================
# FULL MAP PREDICTION
# ============================================================
X_full, y_full = loadData(dataset)
X_full = applyPCA(X_full, PCA_BANDS)
Xp = padWithZeros(X_full, windowSize//2)

out = np.zeros_like(y_full)
coords, patches = [], []

for i in range(y_full.shape[0]):
    for j in range(y_full.shape[1]):
        if y_full[i,j]>0:
            patches.append(Xp[i:i+windowSize,j:j+windowSize])
            coords.append((i,j))

patches = np.array(patches)
preds = np.argmax(model.predict(patches,batch_size=256), axis=1)

for (i,j),p in zip(coords,preds):
    out[i,j]=p+1

spectral.save_rgb(os.path.join(results_folder, model_name+"_map.jpg"),
                  out.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(os.path.join(results_folder, model_name+"_gt.jpg"),
                  y_full.astype(int), colors=spectral.spy_colors)

# ============================================================
# ZIP OUTPUTS
# ============================================================
zip_path = model_name+"_outputs.zip"
with zipfile.ZipFile(zip_path,'w') as z:
    for f in os.listdir(results_folder):
        z.write(os.path.join(results_folder,f), f)
np.save(os.path.join(results_folder, "C-ViT_history.npy"), history.history)
print("✅ ViT experiment completed successfully.")
print("📦 Outputs zipped at:", zip_path)

# ===============HYBRIDSN=================
# Salinas HSI (Memory-Safe, Same Architecture/Hyperparams)
# ================================
# !pip install spectral

import os, time, zipfile, gc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
import spectral
import gc, torch
gc.collect()
tf.keras.backend.clear_session()

# --------- Config ---------
dataset = 'Ho'                 # Salinas
windowSize = 25                # keep as requested
K = 15                        # PCA components
train_ratio = 0.05             # 10% per class for training
batch_size = 256               # keep same as your code
epochs = 100                   # keep same
model_name = "HybridSN"
results_folder = "ulite_results_Ho_per5"
os.makedirs(results_folder, exist_ok=True)

# Optional mixed precision to save memory
mixed_precision.set_global_policy("mixed_float16")
tf.keras.backend.clear_session()
gc.collect()

# --------- Data Loading ---------
def loadData(name):
    # Adjust base path if your files are elsewhere
    base = "/content/drive/MyDrive/Colab Notebooks/dataset/"  # e.g., "/content/drive/MyDrive/Colab Notebooks/dataset/"
    if name == 'Ho':
        data   = sio.loadmat(os.path.join(base, 'Houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(base, 'Houston_gt.mat'))['houston_gt']
    else:
        raise ValueError("Only 'Ho' (Houston) implemented in this script.")
    return data, labels

def applyPCA(X, numComponents):
    Xr = X.reshape(-1, X.shape[2]).astype(np.float32)
    pca = PCA(n_components=numComponents, whiten=True)
    Xp = pca.fit_transform(Xr)
    return Xp.reshape(X.shape[0], X.shape[1], numComponents), pca

def padWithZeros(X, margin=0):
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

# --------- Memory-Safe Patch Generator ---------
class PatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, coords, labels, full_cube, patch_size=25,
                 batch_size=256, shuffle=True, n_classes=20):
        self.coords = coords
        self.labels = labels
        self.full_cube = full_cube
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.half = patch_size // 2
        self.indices = np.arange(len(labels))
        self.padded = padWithZeros(full_cube, self.half)  # pad once
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # preallocate for speed
        batch_X = np.empty((len(idxs), self.patch_size, self.patch_size, self.full_cube.shape[2], 1), dtype=np.float32)
        batch_y = np.empty((len(idxs),), dtype=np.int32)
        for m, k in enumerate(idxs):
            r, c = self.coords[k]
            patch = self.padded[r:r+self.patch_size, c:c+self.patch_size, :]
            batch_X[m, ..., 0] = patch
            batch_y[m] = self.labels[k]
        # NOTE: to_categorical returns float; mixed precision will cast as needed
        batch_y = to_categorical(batch_y, num_classes=self.n_classes)
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --------- Per-Class Split on Coordinates ---------
def splitTrainTestCoords(coords, labels, train_ratio=0.1, seed=42):
    np.random.seed(seed)
    train_idx, test_idx = [], []
    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    return coords[train_idx], coords[test_idx], labels[train_idx], labels[test_idx]

# --------- Build Model (EXACT architecture you used) ---------
def build_model(S, L, num_classes):
    inp = tf.keras.layers.Input((S, S, L, 1))
    x = tf.keras.layers.Conv3D(filters=8,  kernel_size=(3,3,7), activation='relu')(inp)
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,5), activation='relu')(x)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(x)
    # reshape to 2D convs
    conv3d_shape = x.shape  # (None, H, W, D, C)
    x = tf.keras.layers.Reshape((conv3d_shape[1], conv3d_shape[2],
                                 conv3d_shape[3]*conv3d_shape[4]))(x)
    x = tf.keras.layers.Conv2D(filters=64,  kernel_size=(3,3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Conv2D(filters=96,  kernel_size=(3,3), activation='relu')(x)
    #x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    #x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu')(x)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    # Adam 0.001 as in your code
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# =========================
# Pipeline
# =========================
# 1) Load + PCA
X_full, y_full = loadData(dataset)
n_classes = int(y_full.max())  # 16 for Salinas
X_pca, pca = applyPCA(X_full, numComponents=K)

# 2) Build coordinate list (only labeled pixels with border safety)
margin = windowSize // 2
coords, labels = [], []
for r in range(margin, X_pca.shape[0]-margin):
    for c in range(margin, X_pca.shape[1]-margin):
        lab = y_full[r, c]
        if lab > 0:
            coords.append((r, c))
            labels.append(lab - 1)  # zero-based
coords = np.array(coords, dtype=np.int32)
labels = np.array(labels, dtype=np.int32)

# 3) Train/Test split (10% per class)
train_coords, test_coords, ytrain_idx, ytest_idx = splitTrainTestCoords(coords, labels, train_ratio=train_ratio)

# 4) Generators
train_gen = PatchGenerator(train_coords, ytrain_idx, X_pca, patch_size=windowSize,
                           batch_size=batch_size, shuffle=True, n_classes=n_classes)
test_gen  = PatchGenerator(test_coords,  ytest_idx,  X_pca, patch_size=windowSize,
                           batch_size=batch_size, shuffle=False, n_classes=n_classes)

# 5) Model
model = build_model(windowSize, K, n_classes)

# Save model summary
with open(os.path.join(results_folder, f"{model_name}_summary.txt"), "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

# 6) Train
tic = time.perf_counter()
history = model.fit(train_gen, validation_data=test_gen, epochs=epochs, verbose=2)
toc = time.perf_counter()
train_time = toc - tic

# 7) Plots: accuracy & loss
plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Value"); plt.legend()
plt.title("Training vs Validation")
plt.savefig(os.path.join(results_folder, f"{model_name}_training.png"), dpi=150)
plt.close()

# 8) Evaluation on Test Set
tic1 = time.perf_counter()
y_pred_prob = model.predict(test_gen, verbose=0)
toc1 = time.perf_counter()
test_time = toc1 - tic1

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = ytest_idx  # already zero-based labels

# Metrics
classification = classification_report(y_true, y_pred, digits=4)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
oa = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
each_acc = np.nan_to_num(np.diag(cm) / cm.sum(axis=1, keepdims=False))
aa = np.mean(each_acc)
kappa = cohen_kappa_score(y_true, y_pred)

# Confusion matrix figure
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap='viridis')
plt.title('Confusion Matrix'); plt.colorbar()
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, f"{model_name}_confusion_matrix.png"), dpi=150)
plt.close()

# 9) Save Results Text (all metrics + model desc)
with open(os.path.join(results_folder, f"{model_name}_results.txt"), "w") as f:
    f.write(f"Model: 3D→2D CNN (Conv3D(8,3x3x7)->Conv3D(16,3x3x5)->Conv3D(32,3x3x3)"
            f"->Reshape->Conv2D(24,3x3)->Conv2D(96,3x3)->MaxPool2D(2x2)->Conv2D(128,3x3)"
            f"->GAP->Dense(128)->Dropout(0.4)->Dense(64)->Dropout(0.4)->Dense({n_classes}))\n")
    f.write(f"Optimizer: Adam(1e-3), Batch Size: {batch_size}, Epochs: {epochs}\n")
    f.write(f"Window Size: {windowSize}x{windowSize}, PCA Components: {K}\n")
    f.write(f"Training ratio per class: {int(train_ratio*100)}%\n\n")

    f.write(f"Training time: {train_time:.2f} s\n")
    f.write(f"Testing time: {test_time:.2f} s\n\n")

    f.write(f"Overall Accuracy: {oa*100:.2f}%\n")
    f.write(f"Average Accuracy: {aa*100:.2f}%\n")
    f.write(f"Kappa: {kappa*100:.2f}%\n")
    f.write(f"Precision (weighted): {precision*100:.2f}%\n")
    f.write(f"Recall (weighted): {recall*100:.2f}%\n")
    f.write(f"F1-score (weighted): {f1*100:.2f}%\n\n")

    f.write("Classwise Accuracy (%):\n")
    f.write(", ".join([f"{x*100:.2f}" for x in each_acc]) + "\n\n")

    f.write("Classification Report:\n")
    f.write(classification + "\n\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm) + "\n")

# 10) Full Map Prediction (batched, memory-safe)
print("Predicting full map (batched)...")
PATCH = windowSize
pad = PATCH // 2
Xp = padWithZeros(X_pca, pad)
H, W = y_full.shape
outputs = np.zeros((H, W), dtype=np.int32)

# Collect coords of labeled pixels
coords_all = [(r, c) for r in range(pad, H-pad) for c in range(pad, W-pad) if y_full[r, c] > 0]
B = 2048  # batch for full-map inference
for start in range(0, len(coords_all), B):
    batch_coords = coords_all[start:start+B]
    batch = np.empty((len(batch_coords), PATCH, PATCH, K, 1), dtype=np.float32)
    for i, (r, c) in enumerate(batch_coords):
        patch = Xp[r-pad:r+pad+1, c-pad:c+pad+1, :]
        batch[i, ..., 0] = patch
    preds = np.argmax(model.predict(batch, verbose=0), axis=1)
    for (r, c), p in zip(batch_coords, preds):
        outputs[r, c] = p + 1  # back to 1..C

# Save classified and ground-truth maps
spectral.save_rgb(os.path.join(results_folder, f"{model_name}_classified_map.jpg"),
                  outputs.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(os.path.join(results_folder, f"{model_name}_groundtruth.jpg"),
                  y_full, colors=spectral.spy_colors)

# 11) Zip everything
zip_path = f"{model_name}_outputs.zip"
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for fn in os.listdir(results_folder):
        zipf.write(os.path.join(results_folder, fn), arcname=fn)
np.save(os.path.join(results_folder, "hybridsn_history.npy"), history.history)
print(f"✅ Done. All outputs saved in: {zip_path}")

import numpy as np
import matplotlib.pyplot as plt

histories = {
    "HybridSN": np.load("ulite_results_Ho_per5/hybridsn_history.npy", allow_pickle=True).item(),
    "ULite": np.load("ulite_results_Ho_per5/ulite_history.npy", allow_pickle=True).item(),
    "C-ViT": np.load("ulite_results_Ho_per5/C-ViT_history.npy", allow_pickle=True).item(),
    "LhSSN": np.load("ulite_results_Ho_per5/lhSSN_history.npy", allow_pickle=True).item()
}

plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['loss'], label=name)

plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)

plt.savefig("training_loss_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['val_loss'], label=name)

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(True)

plt.savefig("validation_loss_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['val_accuracy'], label=name)

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.savefig("validation_accuracy_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['accuracy'], label=name)

plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.savefig("training_accuracy_comparison.png", dpi=300)
plt.show()

##### Zoomed Graph#####
import numpy as np
import matplotlib.pyplot as plt

# Load histories
histories = {
    "HybridSN": np.load("/content/hybridsn_history.npy", allow_pickle=True).item(),
    "ULite": np.load("/content/ulite_history.npy", allow_pickle=True).item(),
    "C-ViT": np.load("/content/C-ViT_history.npy", allow_pickle=True).item(),
    "LhSSN": np.load("/content/lhSSN_history.npy", allow_pickle=True).item()
}

# ===============================
# 1. Validation Accuracy (Zoomed)
# ===============================
plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['val_accuracy'], linewidth=2, label=name)

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.ylim(0.4, 1.0)   # 👈 KEY CHANGE
plt.legend()
plt.grid(True)
plt.xlim(0, 100)  # focus full epochs
plt.savefig("val_accuracy_zoom.png", dpi=300)
plt.show()


# ===============================
# 2. Validation Loss (Zoomed + Reverse)
# ===============================
plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['val_loss'], linewidth=2, label=name)

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.ylim(0, 1.5)   # 👈 reverse axis (important)
plt.legend()
plt.grid(True)
plt.xlim(0, 100)  # focus full epochs
plt.savefig("val_loss_zoom.png", dpi=300)
plt.show()
# ===============================
# 3. Training Loss (Zoomed + Reverse)
# ===============================

plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['loss'], linewidth=2, label=name)

plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.ylim(0, 1.5)   # 👈 reverse axis (important)
plt.legend()
plt.grid(True)
plt.xlim(0, 100)  # focus full epochs
plt.savefig("training_loss_zoom.png", dpi=300)
plt.show()

# ===============================
# 4. Training Accuracy
# ===============================
plt.figure(figsize=(8,6))

for name, h in histories.items():
    plt.plot(h['accuracy'], linewidth=2, label=name)

plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Comparison")
plt.ylim(0.4, 1.0)   # 👈 KEY CHANGE
plt.legend()
plt.grid(True)
plt.xlim(0, 100)  # focus full epochs
plt.savefig("Training_accuracy_zoom.png", dpi=300)
plt.show()

"""# New section"""
