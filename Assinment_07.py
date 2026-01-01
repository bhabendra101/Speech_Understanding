import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    duration = 0.5  # seconds
    N = int(Fs * duration)
    n = np.arange(N)

    # Frequencies for a major chord
    f_root = f
    f_major_third = f * (2 ** (4 / 12))
    f_major_fifth = f * (2 ** (7 / 12))

    # Convert to angular frequencies
    w1 = 2 * np.pi * f_root / Fs
    w2 = 2 * np.pi * f_major_third / Fs
    w3 = 2 * np.pi * f_major_fifth / Fs

    # Generate chord (sum of tones)
    x = np.cos(w1 * n) + np.cos(w2 * n) + np.cos(w3 * n)

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.exp(-2j * np.pi * k * n / N)
    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    # DFT using FFT (allowed unless explicitly forbidden)
    X = np.fft.fft(x)

    # Magnitude spectrum (positive frequencies only)
    magnitudes = np.abs(X[:N // 2])
    freqs = np.fft.fftfreq(N, d=1/Fs)[:N // 2]

    # Find indices of three largest peaks
    peak_indices = np.argsort(magnitudes)[-3:]

    # Get corresponding frequencies and sort
    loudest_freqs = np.sort(freqs[peak_indices])

    return loudest_freqs[0], loudest_freqs[1], loudest_freqs[2]
