"""ESC-50 four-class audio classification assignment.

The script is self-contained and implements MFCC extraction without librosa.
It creates figures, CSV files, a JSON summary, and a Markdown report under
``assignment/results``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_CLASSES = ("dog", "rain", "crying_baby", "clock_tick")
SAMPLE_RATE = 22050
N_MFCC = 40
RANDOM_STATE = 42


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 64) -> np.ndarray:
    """Create triangular Mel filters for a one-sided power spectrum."""
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1 : index + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_fft // 2)
        for frequency_bin in range(left, min(center, filters.shape[1])):
            filters[index - 1, frequency_bin] = (frequency_bin - left) / (center - left)
        for frequency_bin in range(center, min(right, filters.shape[1])):
            filters[index - 1, frequency_bin] = (right - frequency_bin) / (right - center)
    return filters


def load_audio(path: Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    sample_rate, signal = wavfile.read(path)
    source_dtype = signal.dtype
    signal = signal.astype(np.float32)
    if np.issubdtype(source_dtype, np.integer):
        signal /= np.iinfo(source_dtype).max
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if sample_rate != target_rate:
        divisor = np.gcd(sample_rate, target_rate)
        signal = resample_poly(signal, target_rate // divisor, sample_rate // divisor)
    return signal


def extract_mfcc(path: Path, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return an MFCC matrix with shape (frames, n_mfcc)."""
    signal = load_audio(path)
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    frame_length = int(0.025 * SAMPLE_RATE)
    hop_length = int(0.010 * SAMPLE_RATE)
    n_fft = 1024
    frame_count = 1 + int(np.ceil(max(0, len(signal) - frame_length) / hop_length))
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(signal, (0, max(0, padded_length - len(signal))))
    starts = np.arange(frame_count)[:, None] * hop_length
    offsets = np.arange(frame_length)[None, :]
    frames = padded[starts + offsets] * np.hamming(frame_length)

    power_spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 / n_fft
    mel_energies = power_spectrum @ mel_filterbank(SAMPLE_RATE, n_fft).T
    log_mel = np.log(np.maximum(mel_energies, np.finfo(float).eps))
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def extract_mfcc_summary(path: Path, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Return mean and standard deviation of MFCCs: 2 * n_mfcc values."""
    mfcc = extract_mfcc(path, n_mfcc)
    return np.concatenate((mfcc.mean(axis=0), mfcc.std(axis=0))).astype(np.float32)


def extract_dataset(meta: pd.DataFrame, audio_dir: Path, cache_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_rows = pd.DataFrame(cached["rows"].tolist())
        if cached_rows["filename"].tolist() == meta["filename"].tolist():
            print(f"Using cached MFCC features: {cache_path}")
            return cached["features"], cached["labels"], cached_rows
        print("Selected classes changed; rebuilding the MFCC cache.")

    features = []
    for number, filename in enumerate(meta["filename"], start=1):
        print(f"Extracting MFCC {number:3d}/{len(meta)}: {filename}")
        features.append(extract_mfcc_summary(audio_dir / filename))
    feature_array = np.vstack(features)
    labels = meta["category"].to_numpy()
    rows = meta[["filename", "fold", "category"]].to_dict("records")
    np.savez_compressed(cache_path, features=feature_array, labels=labels, rows=np.array(rows, dtype=object))
    return feature_array, labels, meta[["filename", "fold", "category"]].copy()


def save_pca_plot(points: np.ndarray, labels: np.ndarray, classes: tuple[str, ...], path: Path) -> None:
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    for category in classes:
        mask = labels == category
        axis.scatter(points[mask, 0], points[mask, 1], points[mask, 2], label=category, s=42, alpha=0.8)
    axis.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title="ESC-50 MFCC feature space (PCA 3D)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_data_split_plot(rows: pd.DataFrame, classes: tuple[str, ...], path: Path) -> None:
    counts = pd.crosstab(rows["category"], rows["split"]).reindex(classes)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(counts.index, counts["train"], label="Train (70%)", color="tab:blue")
    axis.bar(counts.index, counts["test"], bottom=counts["train"], label="Test (30%)", color="tab:orange")
    for index, category in enumerate(counts.index):
        axis.text(index, counts.loc[category, "train"] / 2, str(counts.loc[category, "train"]), ha="center", va="center", color="white", fontweight="bold")
        axis.text(index, counts.loc[category, "train"] + counts.loc[category, "test"] / 2, str(counts.loc[category, "test"]), ha="center", va="center", color="white", fontweight="bold")
    axis.set(title="Selected ESC-50 classes and stratified data split", xlabel="Class", ylabel="Number of audio clips", ylim=(0, 45))
    axis.tick_params(axis="x", rotation=15)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_mfcc_plot(meta: pd.DataFrame, audio_dir: Path, classes: tuple[str, ...], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    image = None
    for axis, category in zip(axes.flat, classes):
        filename = meta.loc[meta["category"] == category, "filename"].iloc[0]
        mfcc = extract_mfcc(audio_dir / filename).T
        times = np.arange(mfcc.shape[1]) * 0.010
        image = axis.imshow(mfcc, origin="lower", aspect="auto", cmap="magma", extent=(times[0], times[-1], 1, N_MFCC))
        axis.set(title=f"{category}: {filename}", xlabel="Time (s)", ylabel="MFCC coefficient")
    figure.suptitle("MFCC features for one example from each class", fontsize=15)
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.08, top=0.90, hspace=0.30, wspace=0.18)
    color_axis = figure.add_axes((0.89, 0.16, 0.018, 0.68))
    figure.colorbar(image, cax=color_axis, label="MFCC value")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_pca_variance_plot(cumulative_variance: np.ndarray, components_80: int, path: Path) -> None:
    component_numbers = np.arange(1, len(cumulative_variance) + 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(component_numbers, cumulative_variance, linewidth=2, color="tab:blue")
    axis.axhline(0.80, color="tab:red", linestyle="--", label="80% variance threshold")
    axis.axvline(components_80, color="tab:green", linestyle="--", label=f"{components_80} components")
    axis.scatter(components_80, cumulative_variance[components_80 - 1], color="black", zorder=3)
    axis.annotate(f"{cumulative_variance[components_80 - 1]:.2%}", (components_80, cumulative_variance[components_80 - 1]), xytext=(8, 10), textcoords="offset points")
    axis.set(title="PCA cumulative explained variance", xlabel="Number of principal components", ylabel="Cumulative explained variance", ylim=(0, 1.03))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_k_plot(scores: pd.DataFrame, best_k: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(scores["k"], scores["mean_cv_accuracy"], yerr=scores["std_cv_accuracy"], marker="o", capsize=3)
    axis.axvline(best_k, color="tab:red", linestyle="--", label=f"best k = {best_k}")
    axis.set(xlabel="Number of neighbors (k)", ylabel="5-fold CV accuracy", title="kNN performance for different k values")
    axis.set_xticks(scores["k"])
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_confusion_matrices(y_test: np.ndarray, predictions: dict[str, np.ndarray], classes: tuple[str, ...], path: Path) -> None:
    figure, axes = plt.subplots(1, len(predictions), figsize=(13, 5))
    for axis, (name, prediction) in zip(np.atleast_1d(axes), predictions.items()):
        matrix = confusion_matrix(y_test, prediction, labels=classes)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, ax=axis)
        axis.set(title=f"{name} confusion matrix", xlabel="Predicted", ylabel="True")
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_model_comparison(accuracies: dict[str, float], path: Path) -> None:
    names = list(accuracies)
    values = [accuracies[name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(names, values, color=["tab:blue", "tab:orange"], width=0.55)
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=4, fontsize=12)
    axis.set(title="Test accuracy comparison: kNN vs RBF-SVM", xlabel="Classifier", ylabel="Test accuracy", ylim=(0, 1.0))
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(output_dir: Path, summary: dict, classes: tuple[str, ...]) -> None:
    report = f"""# ESC-50 四分类音频分类实验报告

## 1. 实验任务

从 ESC-50 中选择 `{', '.join(classes)}` 四类环境声音，每类 40 条，共 160 条。数据使用固定随机种子 {RANDOM_STATE} 进行分层划分：70% 训练集（112 条）和 30% 测试集（48 条），因此每类训练 28 条、测试 12 条。

![选取类别与数据划分](./data_split.png)

## 2. 特征提取

音频统一重采样到 {SAMPLE_RATE} Hz。处理步骤为：预加重、25 ms 分帧、10 ms 帧移、Hamming 窗、功率谱、64 个 Mel 滤波器、取对数、DCT。每帧保留 {N_MFCC} 个 MFCC；再对时间维分别计算均值和标准差，因此每段音频得到 **{2 * N_MFCC} 维**特征。

下图分别展示四个类别中一段代表音频的 MFCC。横轴是时间，纵轴是 MFCC 系数编号，颜色表示系数值。

![四类音频的 MFCC 特征](./mfcc_features.png)

## 3. PCA 分析

PCA 只在标准化后的训练数据上拟合，再用于训练集和测试集，避免数据泄漏。累计解释方差达到至少 80% 所需的主成分数为 **{summary['pca_components_80']}**，实际累计解释方差为 **{summary['pca_variance_80']:.2%}**。三维特征空间见 `pca_3d.png`。

![PCA 三维特征空间](./pca_3d.png)

![PCA 累计解释方差](./pca_cumulative_variance.png)

## 4. kNN 实验

在训练集上使用 5 折分层交叉验证比较 k=1 至 20，最优值为 **k={summary['best_k']}**，平均交叉验证准确率为 **{summary['best_knn_cv_accuracy']:.2%}**。最后仅在选定 k 后评估测试集，测试准确率为 **{summary['knn_test_accuracy']:.2%}**。调参曲线见 `knn_k_performance.png`。

![不同 k 值的 kNN 交叉验证准确率](./knn_k_performance.png)

## 5. 第二个分类器：SVM

使用 RBF 核 SVM（标准化、C=10、gamma=`scale`）在同一训练集上训练，在同一测试集上的准确率为 **{summary['svm_test_accuracy']:.2%}**。

## 6. 结果比较与结论

| 模型 | 测试准确率 |
|---|---:|
| kNN (k={summary['best_k']}) | {summary['knn_test_accuracy']:.2%} |
| RBF-SVM | {summary['svm_test_accuracy']:.2%} |

![kNN 与 SVM 测试准确率对比](./model_comparison.png)

本实验中测试集表现更好的模型是 **{summary['better_model']}**。混淆矩阵见 `confusion_matrices.png`，逐类别 precision、recall 和 F1-score 见 `classification_report.csv`。由于总样本量只有 160，单次 70/30 划分的结果存在随机波动；训练阶段使用交叉验证选择 k，测试集只用于最终比较。

![kNN 与 SVM 混淆矩阵](./confusion_matrices.png)

## 7. 复现实验

```bash
python assignment/esc50_audio_classification.py
```
"""
    (output_dir / "report_zh.md").write_text(report, encoding="utf-8")


def run(project_root: Path, classes: tuple[str, ...], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(project_root / "meta" / "esc50.csv")
    selected = metadata[metadata["category"].isin(classes)].copy().reset_index(drop=True)
    counts = selected["category"].value_counts()
    if len(selected) != 40 * len(classes) or not (counts == 40).all():
        raise ValueError(f"Expected 40 files per class, found: {counts.to_dict()}")

    cache_path = output_dir / "mfcc_features.npz"
    features, labels, rows = extract_dataset(selected, project_root / "audio", cache_path)
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)), test_size=0.30, stratify=labels, random_state=RANDOM_STATE
    )
    x_train, x_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    rows["split"] = ""
    rows.loc[train_indices, "split"] = "train"
    rows.loc[test_indices, "split"] = "test"
    rows.to_csv(output_dir / "data_split.csv", index=False)
    save_data_split_plot(rows, classes, output_dir / "data_split.png")
    save_mfcc_plot(selected, project_root / "audio", classes, output_dir / "mfcc_features.png")

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    pca = PCA().fit(x_train_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components_80 = int(np.searchsorted(cumulative_variance, 0.80) + 1)
    save_pca_variance_plot(cumulative_variance, components_80, output_dir / "pca_cumulative_variance.png")
    pca_3d = PCA(n_components=3).fit(x_train_scaled)
    all_points = pca_3d.transform(scaler.transform(features))
    save_pca_plot(all_points, labels, classes, output_dir / "pca_3d.png")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    k_rows = []
    for k in range(1, 21):
        pipeline = Pipeline((("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))))
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="accuracy")
        k_rows.append({"k": k, "mean_cv_accuracy": scores.mean(), "std_cv_accuracy": scores.std()})
    k_scores = pd.DataFrame(k_rows)
    best_row = k_scores.sort_values(["mean_cv_accuracy", "k"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["k"])
    k_scores.to_csv(output_dir / "knn_k_scores.csv", index=False)
    save_k_plot(k_scores, best_k, output_dir / "knn_k_performance.png")

    knn = Pipeline((("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k))))
    svm = Pipeline((("scale", StandardScaler()), ("svm", SVC(kernel="rbf", C=10, gamma="scale"))))
    knn.fit(x_train, y_train)
    svm.fit(x_train, y_train)
    predictions = {"kNN": knn.predict(x_test), "RBF-SVM": svm.predict(x_test)}
    accuracies = {name: accuracy_score(y_test, prediction) for name, prediction in predictions.items()}
    save_model_comparison(accuracies, output_dir / "model_comparison.png")
    save_confusion_matrices(y_test, predictions, classes, output_dir / "confusion_matrices.png")

    reports = []
    for model_name, prediction in predictions.items():
        frame = pd.DataFrame(classification_report(y_test, prediction, labels=classes, output_dict=True, zero_division=0)).T
        frame.insert(0, "model", model_name)
        frame.insert(1, "label", frame.index)
        reports.append(frame.reset_index(drop=True))
    pd.concat(reports, ignore_index=True).to_csv(output_dir / "classification_report.csv", index=False)

    summary = {
        "classes": list(classes),
        "feature_dimension": int(features.shape[1]),
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "pca_components_80": components_80,
        "pca_variance_80": float(cumulative_variance[components_80 - 1]),
        "best_k": best_k,
        "best_knn_cv_accuracy": float(best_row["mean_cv_accuracy"]),
        "knn_test_accuracy": float(accuracies["kNN"]),
        "svm_test_accuracy": float(accuracies["RBF-SVM"]),
        "better_model": max(accuracies, key=accuracies.get),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, summary, classes)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs=4, default=DEFAULT_CLASSES, metavar="CLASS")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    root = Path(__file__).resolve().parents[1]
    run(root, tuple(arguments.classes), arguments.output_dir.resolve())
