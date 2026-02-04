import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    '''
    # Number of frames
    nframes = 1 + (len(speech) - frame_length) // frame_skip
    
    # Output arrays
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]
        
        # Apply window
        frame = frame * np.hamming(frame_length)
        
        # LPC coefficients
        a = librosa.lpc(frame, order)
        A[i, :] = a
        
        # Compute excitation (prediction error)
        e = np.zeros(frame_length)
        for n in range(order, frame_length):
            e[n] = frame[n] - np.sum(a[1:] * frame[n-1:n-order-1:-1])
        
        excitation[i, :] = e
    
    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    '''
    nframes, frame_length = e.shape
    order = A.shape[1] - 1
    
    duration = nframes * frame_skip
    synthesis = np.zeros(duration)
    
    for i in range(nframes):
        start = i * frame_skip
        a = A[i]
        
        for n in range(frame_skip):
            idx = start + n
            if idx < duration:
                synthesis[idx] = e[i, n]
                for k in range(1, order + 1):
                    if idx - k >= 0:
                        synthesis[idx] -= a[k] * synthesis[idx - k]
    
    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Create robot voice excitation.
    '''
    nframes, frame_length = excitation.shape
    
    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)
    
    for i in range(nframes):
        # Gain = RMS of excitation frame
        gain[i] = np.sqrt(np.mean(excitation[i]**2))
        
        # Create impulse train
        for n in range(frame_skip):
            if (i * frame_skip + n) % T0 == 0:
                e_robot[i * frame_skip + n] = gain[i]
    
    return gain, e_robot