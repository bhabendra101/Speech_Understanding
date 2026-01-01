import numpy as np
import matplotlib.pyplot as plt

def center_of_gravity(x):
    '''
    Find the center of gravity of a vector, x.
    '''
    indices = np.arange(len(x))
    c = np.dot(indices, x) / np.sum(x)
    return c

def matched_identity(x):
    '''
    Create an identity matrix that has the same number of rows as x has elements.
    '''
    I = np.eye(len(x))
    return I

def sine_and_cosine(t_start, t_end, t_steps):
    '''
    Create a time axis, and compute its cosine and sine.
    '''
    t = np.linspace(t_start, t_end, t_steps)
    x = np.cos(t)
    y = np.sin(t)
    return t, x, y
