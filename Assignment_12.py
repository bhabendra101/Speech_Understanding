import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    
    excitation[n] = -1 if n is an integer multiple of round(Fs/F0)
    excitation[n] = 0 otherwise
    '''
    excitation = np.zeros(duration)
    
    # Pitch period in samples
    period = int(np.round(Fs / F0))
    
    # Generate impulse train
    for n in range(0, duration, period):
        excitation[n] = -1
    
    return excitation


def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    Second-order IIR filter implementation.
    '''
    N = len(x)
    y = np.zeros(N)
    
    # Pole radius and angle
    r = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs
    
    # Filter coefficients
    a1 = 2 * r * np.cos(theta)
    a2 = -r**2
    
    # Difference equation
    for n in range(2, N):
        y[n] = x[n] + a1 * y[n-1] + a2 * y[n-2]
    
    return y


def synthesize_vowel(duration, F0, F1, F2, F3, F4,
                     BW1, BW2, BW3, BW4, Fs):
    '''
    Synthesize a vowel using cascade of formant resonators.
    '''
    # Generate voiced excitation
    excitation = voiced_excitation(duration, F0, Fs)
    
    # Cascade of four formant resonators
    y1 = resonator(excitation, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    y4 = resonator(y3, F4, BW4, Fs)
    
    return y4