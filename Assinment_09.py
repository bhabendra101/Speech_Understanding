import numpy as np

# ---------- Helper functions ----------
def pre_emphasize(x, alpha=0.97):
    return np.append(x[0], x[1:] - alpha * x[:-1])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- Main functions ----------
def VAD(waveform, Fs):
    '''
    Voice Activity Detection using short-time energy.
    '''
    frame_length = int(0.025 * Fs)  # 25 ms
    step = int(0.010 * Fs)          # 10 ms

    # Frame the signal
    num_frames = 1 + (len(waveform) - frame_length) // step
    energies = []
    frames = []

    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length]
        energy = np.sum(frame ** 2)

        frames.append(frame)
        energies.append(energy)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    # Extract voiced segments
    segments = []
    current_segment = []

    for i, energy in enumerate(energies):
        if energy > threshold:
            current_segment.extend(frames[i])
        else:
            if len(current_segment) > 0:
                segments.append(np.array(current_segment))
                current_segment = []

    if len(current_segment) > 0:
        segments.append(np.array(current_segment))

    return segments


def segments_to_models(segments, Fs):
    '''
    Create average log-spectrum models from speech segments.
    '''
    models = []

    frame_length = int(0.004 * Fs)  # 4 ms
    step = int(0.002 * Fs)          # 2 ms

    for segment in segments:
        # Pre-emphasis
        segment = pre_emphasize(segment)

        # Frame the segment
        num_frames = 1 + (len(segment) - frame_length) // step
        spectra = []

        for i in range(num_frames):
            start = i * step
            frame = segment[start:start + frame_length]

            # FFT magnitude
            spectrum = np.abs(np.fft.fft(frame))

            # Keep low-frequency half
            spectrum = spectrum[:len(spectrum)//2]

            spectra.append(spectrum)

        spectra = np.array(spectra)

        # Average log spectrum
        log_spectrum = 20 * np.log10(np.maximum(spectra, 1e-10))
        model = np.mean(log_spectrum, axis=0)

        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity.
    '''
    # Extract test segments
    test_segments = VAD(testspeech, Fs)

    # Convert test segments to models
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):
        for y, model in enumerate(models):
            sims[y, k] = cosine_similarity(test_model, model)

        # Best matching model
        best_index = np.argmax(sims[:, k])
        test_outputs.append(labels[best_index])

    return sims, test_outputs
