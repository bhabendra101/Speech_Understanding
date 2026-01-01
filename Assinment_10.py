import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------- helper functions ----------
def pre_emphasize(x, alpha=0.97):
    return np.append(x[0], x[1:] - alpha * x[:-1])


def frame_signal(x, frame_length, step):
    num_frames = 1 + (len(x) - frame_length) // step
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * step
        frames[i] = x[start:start + frame_length]
    return frames


# ---------- main functions ----------
def get_features(waveform, Fs):
    '''
    Extract features and labels from a waveform.
    '''
    # -------- FEATURE EXTRACTION --------
    waveform = pre_emphasize(waveform)

    frame_len_feat = int(0.004 * Fs)  # 4 ms
    step_feat = int(0.002 * Fs)       # 2 ms

    frames = frame_signal(waveform, frame_len_feat, step_feat)
    stft = np.abs(np.fft.fft(frames, axis=1))

    # Keep non-aliased (low-frequency) half
    features = stft[:, :stft.shape[1] // 2]

    # -------- VAD FOR LABELS --------
    frame_len_vad = int(0.025 * Fs)  # 25 ms
    step_vad = int(0.010 * Fs)       # 10 ms

    vad_frames = frame_signal(waveform, frame_len_vad, step_vad)
    energies = np.sum(vad_frames ** 2, axis=1)
    threshold = 0.1 * np.max(energies)

    labels = np.zeros(len(features), dtype=int)
    current_label = 0
    frame_ratio = int(step_vad / step_feat)

    i = 0
    for idx, energy in enumerate(energies):
        if energy > threshold:
            current_label += 1
            for _ in range(5):  # repeat label 5 times
                if i < len(labels):
                    labels[i] = current_label
                    i += 1
        else:
            i += frame_ratio

    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    Train a neural network using PyTorch.
    '''
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    n_feats = features.shape[1]
    n_labels = int(labels.max()) + 1

    model = nn.Sequential(
        nn.LayerNorm(n_feats),
        nn.Linear(n_feats, n_labels)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Run inference and return softmax probabilities.
    '''
    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.detach().numpy()
