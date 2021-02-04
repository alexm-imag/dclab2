# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:08:43 2021

@author: Alexander Müller
"""

# %% Library import
# figures inline          
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp              

# %% Definitions (yeah I like C coding...)

UNIPOLAR = 0
BIPOLAR = 1

RECT_PULSE = 0
RRC_PULSE = 1

# %% Setting up parameters
N = 10000
us_factor=8
mod_type = UNIPOLAR
pulse_shape = RRC_PULSE
alpha = 0.5

bin_seq = np.random.randint(2, size=N)
t = np.arange(0, N)
t_up = np.arange(0, N * us_factor)

Eb_N0 = 10

if N > 100:
    NOPLOTS = 1
else:
    NOPLOTS = 0

# %% Functions
def my_plot(t, data, stem, yaxis, title):
    
    if(NOPLOTS == 1):
        return
    
    plt.figure(figsize=(7,2))
    plt.yticks(yaxis)
    if(stem > 0):
        plt.stem(t, data)
    else:
        plt.plot(t, data)
        
    plt.title(title)
    plt.show()


def is_singular(x):
    """
    Check if x is close to zero within numpy's numerical precision for float

    Parameters:
    -----------
    x : array-like
    """
    return np.abs(x) < np.finfo(np.float).eps / 2.0

def rrc(t, alpha, T=1):
    """
    Generates a root-raised-cosine (RRC) FIR filter impulse response.
    Parameters
    ----------
    t : 1-D array
        Time support array. May also be integer-valued.
    alpha : float
        Roll off factor  should be within [0, 1].
    T : scalar
        Symbol period either in seconds, if t is real-valued, or as an
        integer number of samples per symbol, if t is integer-valued.
    Returns
    ---------
    h : 1-D ndarray of floats
        Unit-energy impulse response of the root raised cosine filter, i.e.
        the impulse response is normalized such that its energy amounts to 1.
    """
    tn = t/T  # normalized time array
    a = alpha  # to shorten the code lines below
    is_zero = is_singular(tn)  # logic array that is true where t==0
    tn[is_zero] = 1  # set t to dummy value at those positions
    # logic array that is true, where t = +-Ts/(4*alpha)
    singularities = is_singular(abs(tn) - 1/(4*a))
    tn[singularities] = 1  # set t to dummy value at those positions
    # Now computing h for all positions should be safe
    h = ((np.sin(np.pi*tn*(1-a)) + 4*a*tn*np.cos(np.pi*tn*(1+a)))
         / (np.pi*tn*(1-(4*a*tn)**2))) / np.sqrt(T)
    # Handle t=0
    h[is_zero] = (1.0 - a + (4*a/np.pi)) / np.sqrt(T)
    # and now the sinularities at t = +-Ts/(4*alpha)
    h[singularities] = a/np.sqrt(2*T)*((1+2/np.pi)*np.sin(np.pi/(4*a))
                                       + (1-2/np.pi)*np.cos(np.pi/(4*a)))
    return h

# %% filter coefficients
h_rect = np.ones(us_factor)     
t_rrc = np.arange(-N/2, N/2)
h_rrc = rrc(t_rrc, alpha, us_factor)

#my_plot(t_rrc, h_rrc, 0, [min(h_rrc), max(h_rrc)], "h_rrc")

# %% modulation
if(mod_type == UNIPOLAR):
    mod_data = bin_seq
    
if(mod_type == BIPOLAR):
    mod_data = bin_seq  + (bin_seq % 2) - 1
 
my_plot(t, mod_data, 1, [-1, 0 , 1], "data after mod")    

# %% Upsampling
upsampled_data = np.zeros(us_factor * N)
upsampled_data[0::us_factor] = mod_data[::]


my_plot(t_up, upsampled_data, 1, [-1, 0 , 1], "data after upsampling")    

# %% Pulse Shaping

if(pulse_shape == RECT_PULSE):
    s = np.convolve(upsampled_data, h_rect, mode='same')
    
if(pulse_shape == RRC_PULSE):
    s = np.convolve(upsampled_data, h_rrc, mode='same')
    
my_plot(t_up, s, 0, [1, 0, -1], "data after pulse shaping")    

# %% channel
test = ( 10*np.log10(Eb_N0))
noise = np.random.normal(size = N * us_factor) * (1 / test)
r = np.add(s,noise)
#r=s
# %% matched filter
if(pulse_shape == RECT_PULSE):
    x = np.convolve(r, h_rect, mode='same')
    
if(pulse_shape == RRC_PULSE):
    x = np.convolve(r, h_rrc, mode='same')

my_plot(t_up, x, 0, [min(x), max(x)], "data after mf")   

# normalize x
xnorm = x / max(x)

# %% down sampling
z = xnorm[0::us_factor]

my_plot(t, z, 1, [min(z), max(z)], "data after downsampling")   

# %% decision
# decide using euclidean distance: min{sqrt((1 - z)^2),sqrt((0 - z)^2)}
dec = np.zeros(N)

if(mod_type == UNIPOLAR):
    for i in range(N):
        if np.sqrt((1 - z[i])**2) < np.sqrt((0 - z[i])**2):
            dec[i] = 1
        else:
            dec[i] = 0
            

if(mod_type == BIPOLAR):
    for i in range(N):
        if np.sqrt((1 - z[i])**2) < np.sqrt((-1 - z[i])**2):
            dec[i] = 1
        else:
            dec[i] = 0
            


my_plot(t, dec, 1, [min(dec), max(dec)], "data after decision")   
my_plot(t, bin_seq, 1, [0 , 1], "generated bin_seq")    

# %% performance
err = 0
for i in range(N):
    if dec[i] != bin_seq[i]:
        err = err +1
        
BER = err / N        

print("sequence length:", N)
print("Eb_N0: ", Eb_N0)
print("errors: ", err)
print("BER: ", BER)

# %% plots
c = round(N/us_factor)

if(NOPLOTS == 0):
    for i in range(c):
        plt.plot(x[i * us_factor ::(i+1) * us_factor])
         
    plt.xlim(0, us_factor )
    plt.ylim(min(x), max(x))
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.title('eye diagram')
    plt.grid(True)
    plt.show()
 

    





