import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    '''
    N = len(waveform)

    # Number of frames
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    '''
    # FFT along each row
    stft = np.fft.fft(frames, axis=1)
    mstft = np.abs(stft)

    return mstft


def mstft_to_spectrogram(mstft):
    '''
    Convert magnitude STFT to decibels with dynamic range limiting.
    '''
    # Floor to avoid log of zero
    max_val = np.amax(mstft)
    mstft_safe = np.maximum(0.001 * max_val, mstft)

    # Convert to decibels
    spectrogram = 20 * np.log10(mstft_safe)

    return spectrogram
