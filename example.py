"""
Simple example of a generated time series with known coefficients that 
is used to feed to pressure time series to the quaternion_mode function
to show how the coefficients are retrieved from the pressure series only
Expression for the azimuthal pressure mode is taken from [1]

References
----------
    [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of 
        azimuthal instabilities", Physical Review Fluids, 2018
"""

import numpy as np
import matplotlib.pyplot as plt

from quatazim import quaternion_mode, filter_signal

# Set sampling time, total runtime and create the time series
fs = 50e3
t_end = 10
time = np.arange(t_end * int(fs)) / fs

# Setting the oscillation frequency
freq = 1200
omega = 2 * np.pi * freq

# Set up the parameters
A = np.ones_like(time) + 0.3 * np.sin(3 * time) * np.exp(-time/4)
# Make the nature angle slowly varying
# Note this should have magnitude of pi/4 or less to make physical sense
chi = 0.9 * np.pi / 4 * np.sin(1.5 * time)
# Create a linear drift in the orientation angle
# for half of the runtime
ntheta_0 = ((time * 2 + np.pi) % (2*np.pi)) - np.pi
t_half = t_end / 2.0
ntheta_0_final = ntheta_0[np.argmin(np.absolute(time - t_half))]
ntheta_0[time > np.mean(time)] = ntheta_0_final
# Fast temporal phase
phi = omega * time

# Set the azimuthal order of the mode
n = 1

# Create a function to return the pressure at a given azimuthal location
def pressure_expr(angle):
    theta = np.deg2rad(angle)
    return (A * np.cos(n * theta - ntheta_0) * np.cos(chi) * np.cos(phi) +
            A * np.sin(n * theta - ntheta_0) * np.sin(chi) * np.sin(phi))

# Setting up the locations to sample the signal
angles = np.asarray([0, 30, 60, 120, 210, 300])
# Create the time series for the different locations
pressures = np.zeros((len(angles), len(time)), dtype=np.float64)
for ind, angle in enumerate(angles):
    pressures[ind, :] = pressure_expr(angle)

# Filter the signal just to be consistent with how the
# function would be used for actual experimental data
# The width of the filtering should be adjusted to
# ensure the full features of the mode is retained, but
# narrow enough to only include the desired mode
dfreq = 50
pressures = filter_signal(pressures, fs, f_low=freq-dfreq, f_high=freq+dfreq)

# Obtain the mode from the mapping to the quaternion variables.
# This may give a warning message that one point is differing,
# but this is due to edge effects of the Hilbert transform
mode = quaternion_mode(angles, pressures, n=n)

# Plot a comparison between the known parameters and the
# ones obtained from the mapping
fig, axes = plt.subplots(2, 2)

def plot_comparison(ax, exact, reconstructed, label):
    ax.plot(time, exact, 'C0', label='Exact', lw=5)
    ax.plot(time, reconstructed, 'C1', label='Reconstructed', lw=2)
    ax.set_ylabel(label)
plot_comparison(axes[0, 0], A, mode['amplitude'], 'Amplitude')
plot_comparison(axes[0, 1], chi, mode['chi'], 'Nature angle')
plot_comparison(axes[1, 0], ntheta_0, mode['ntheta_0'], 'Orientation angle')
plot_comparison(axes[1, 1], phi, np.unwrap(mode['phi']), 'Temporal phase')

# Clean up the axis a bit
for ax in axes[:, -1]:
    ax.get_yaxis().set_label_position('right')
    ax.get_yaxis().set_ticks_position('right')
for ax in axes[0, :]:
    ax.set_xticklabels([])
for ax in axes[1, :]:
    ax.set_xlabel('Time in s')
axes[1, 1].legend(loc='upper left')

plt.show()
plt.close(fig)
