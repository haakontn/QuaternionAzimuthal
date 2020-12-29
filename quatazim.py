from collections import OrderedDict
import warnings

import numpy as np
import scipy.signal
from scipy.fft import next_fast_len


def get_analytical_signal(sig, axis=-1):
    """
    Compute the analytical signal using the Hilbert transform.

    Wrapper for the scipy.signal.hilbert function. Performs 
    the transform over the last axis by default.

    Parameters
    ----------
    sig : array_like
        Real valued signal.
    axis : int, optional
        Axis to perform the transformation over. Default: -1.

    Returns
    -------
    sig_a : ndarray
            Analytical signal of 'sig' along the axis 'axis'
    """
    sig = np.asarray(sig)
    # Find the number of steps in the FFT used for the Hilbert transform
    # This ensures the FFT is as fast as possible in the transform
    # This is the default behaviour for newer versions of ScipPy
    nfft = next_fast_len(sig.shape[axis])
    return scipy.signal.hilbert(sig, axis=axis, N=nfft)


def get_xis(angles, pressures, n=1):
    """
    Calculates the analytical xi values [1].

    Uses the pressure times series and angles to map the 
    pressure time series into the quaternion formalism
    for azimuthal pressure modes in annular configurations [1].
    Using a least squares solver to map the pressure signals
    into the corresponding quaternion parameters. Note the 
    highest order of the azimuthal instability that is 
    possible to resolve is at least one less than the 
    number of angles (depending on how the angles are
    distributed).

    Parameters
    ----------
    angles : array_like
        The azimuthal location of the pressure measurements
        given in degrees for each of the N pressure time
        series in 'pressures'.
        Shape: (N,)
    pressures : array_like
        2D array of the pressure time series, corresponding
        to the azimuthal locations given by angles. The 
        azimuthal location is given by the first dimension,
        and the time steps are given by the second dimension.
        Has to be real valued.
        Shape: (N, M)
    n : int, optional
        Azimuthal order of the mode. Default: 1.

    Returns
    -------
    xis : ndarray
        The two real valued xi series calculated based on [1].
        First row is xi_1, second row is xi_2.
        Shape: (2, M)

    References
    ----------
      [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of 
          azimuthal instabilities", Physical Review Fluids, 2018
    """
    # Make sure the input are ndarrays for easier handling, 
    # alternatively convert them to ndarrays
    angles = np.asarray(angles)
    pressures = np.asarray(pressures)

    # Convert angles to radians and create the least squares matrix
    theta = np.deg2rad(angles)
    mat = np.asarray([np.cos(n*theta), np.sin(n*theta)]).T

    # Calculate the two xi values, [xi_1, xi_2]
    return np.linalg.lstsq(mat, pressures, rcond=None)[0]


def quaternion_mode(angles, pressures, n=1, pi_4_tol=1e-8, amp_tol=1e-8):
    """ 
    Calculates the quaternion mode parameters based on pressure time series.

    Calculating the pressure mode parameters based on the quaternion 
    formalism for annular instabilities, as introduced by Ghirardo and 
    Bothien [1].
    
    Parameters
    ----------
    angles : array_like
        Azimuthal location of the pressure measurements given in degrees. 
        Each of the N elements corresponds to the location of the pressure
        time series in the same row of the 'pressures' array
        Shape: (N,)
    pressures : array_like
        Pressure time series for the different azimuthal locations. The N
        rows correspond to the N azimuthal locations. The M columns 
        correspond to the M points in time the pressure is sampled at.
        Shape: (N, M)
    n : int, optional
        Azimuthal order of the mode. Default: 1.
    pi_4_tol : float, optional
        Tolerance for how close to +-pi/4 is considered pi/4. Default: 1e-8
    amp_tol : float, optional
        Tolerance for how close the amplitudes have to be in the 
        reconstruction check to be considered fine. If it is above this
        threshold value a warning is triggered. Default: 1e-8

    Returns
    -------
    result : OrderedDict
        Dictionary containing the time series for the following parameters
            'amplitude' : Amplitude A of the fluctuations
            'chi'       : Nature angle
            'ntheta_0'  : Orientation angle (n*theta_0)
            'phi'       : Temporal phase (fast)

    Notes
    -----
    This function is not conditioning on the frequency at all,
    and to obtain the desired frequency components of the signal the 
    pressure time series should be band pass filtered around the 
    desired frequency. This should also be done if there is a single
    dominant mode on the pressure signals to avoid fitting the
    low frequency noise as well as the desired pressure fluctuations.

    The Hilbert transform will have some end effects, so in both end
    of the time series the reconstruction might not be perfectly 
    representative of the actual pressure time series. Therefore,
    it can be a good idea to discard the first and last part of the
    reconstructed quaternion mode parameters.

    References
    ----------
      [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of 
          azimuthal instabilities", Physical Review Fluids, 2018
    """

    angles = np.asarray(angles)
    pressures = np.asarray(pressures)
    if len(angles.shape) != 1:
        raise ValueError("expected 'angles' to be 1D array")
    if len(pressures.shape) != 2:
        raise ValueError("expected 'pressures' to be a 2D array")
    if angles.shape[0] != pressures.shape[0]:
        raise ValueError("first dimension of 'angles' and 'pressures' should be equal")

    # Obtain the two real valued xi parameters
    xi_1, xi_2 = get_xis(angles, pressures, n=n)

    # Obtain the analytical signal for the xi parameters
    xia_1 = get_analytical_signal(xi_1)
    xia_2 = get_analytical_signal(xi_2)

    # Find the variables to store, see article Appendix C

    # To avoid the use of external quaternion libraries, 
    # save the quaternion valued analtyical xi values in
    # a (4, M) shaped array called xia.
    # First index corresponds to the following:
    #   0 = real part
    #   1 = i imaginary part
    #   2 = j imaginary part
    #   4 = k imaginary part
    xia = np.asarray([np.real(xia_1), np.real(xia_2), 
                      np.imag(xia_1), np.imag(xia_2)])

    # Create a dictionary to save the result in
    result = OrderedDict()

    # Find amplitude and nature angle
    result['amplitude'] = get_amplitude(xia)
    result['chi'] = get_chi(xia)

    # Normalize the signal for the rest of the calculations
    xia_norm = xia / result['amplitude'][None, :]

    # Shorthand for the nature angle
    chi = result['chi']

    # Allocate space for the orientation angle ntheta_0 and the 
    # temporary temporal phase phi_tmp
    ntheta_0 = np.zeros(len(chi))
    phi_tmp = np.zeros(len(chi))

    # Two cases needs to be handled separately
    #       Case 1) : chi = +- pi/4  
    #       Case 2) : chi != +- pi/4

    # Case 1)
    pi4 = (np.absolute(np.pi - np.absolute(chi)) < pi_4_tol)
    # Quaternion number z is given by z = a + i*b + j*c + k*d
    a, b, c, d = xia_norm[:, pi4]
    phi_tmp[pi4] = np.arctan2(2*(b*d - a*c), a**2 - b**2 - c**2 + d**2) / 2

    # Case 2)
    not_pi4 = np.invert(pi4)
    a, b, c, d = xia_norm[:, not_pi4]
    ntheta_0[not_pi4] = np.arctan2(2*(a*b + c*d), a**2 - b**2 + c**2 - d**2) / 2
    phi_tmp[not_pi4] = np.arctan2(2*(b*d + a*c), a**2 + b**2 - c**2 - d**2) / 2


    # Do some extra steps to make the orientation angle span (-pi, pi]
    # instead of just spanning (-pi/2, pi/2]
    # This has to be done sequentially!
    pi_2 = np.pi / 2
    for ind in range(1, len(ntheta_0)):
        diff = ntheta_0[ind] - ntheta_0[ind-1]
        if np.absolute(diff) > pi_2:
            # If the difference is above pi/2, move the orientation
            # angle an angle pi to bring it to the correct side of 
            # the Poincare sphere
            theta_tmp = ntheta_0[ind] - np.pi * np.sign(diff)
            # In case the new value is above pi, bring it back into
            # the desired interval again
            ntheta_0[ind] = ((theta_tmp + np.pi) % (2 * np.pi)) - np.pi

    # Now the orientation angle ntheta_0 is set correctly
    result['ntheta_0'] = ntheta_0

    # Temporal phase still needs some small corrections
    # Need to check if we are in the correct half plane first
    # Therefore, calculate a, b, c and d from the current values
    # of the quaternion expression to compare with the values
    # obtained from the xia values calculated from the pressure series

    # Get complex valued exponentials
    exp_i = np.exp(1j * ntheta_0)
    exp_k = np.exp(-1j * chi)
    exp_j = np.exp(1j * phi_tmp)

    # Separate the exponentials into real and imaginary part
    ri, ii = exp_i.real, exp_i.imag
    rk, ik = exp_k.real, exp_k.imag
    rj, ij = exp_j.real, exp_j.imag

    # Calculate a, b, c, d based on the real and imaginary parts
    a_ref = ri * rk * rj + ii * ik * ij
    b_ref = ii * rk * rj - ri * ik * ij
    c_ref = ri * rk * ij - ii * ik * rj
    d_ref = ri * ik * rj + ii * rk * ij

    # First check the magnitude to be similar for the reconstructed points
    mag_check = np.absolute([a/a_ref, b/b_ref, c/c_ref, d/d_ref]) - 1
    if np.amax(np.absolute(mag_check)) > amp_tol:
        # Number of points that fail the check
        nfailed = np.sum(np.absolute(mag_check) > amp_tol)
        wrn_msg = ('The control expression differ from the data expression '
                   'for some values. Number of differing elements: {:d}/{:d}'
                   ''.format(nfailed, mag_check.shape[-1]))
        warnings.warn(wrn_msg, stacklevel=2)
    
    # Make space for the final temporal phase by making a copy
    # This way only the points where the temporal phase is wrong
    # needs to be updated
    phi = np.copy(phi_tmp)

    # When the amplitude ratio is negative, the temporal phase has 
    # been shifted by +- pi
    wrong_pos = np.where(np.sign(a / a_ref) < 0)[0]

    # Get the existing phase at the wrong points
    phi_tmp_wrong = phi_tmp[wrong_pos]
    # Set all the corrections to -pi first
    dphi = np.zeros(len(wrong_pos)) - np.pi
    # Where the phase was less than 0, the phase correction
    # should be +pi to bring it into the (-pi, pi] interval
    dphi[phi_tmp_wrong < 0] = np.pi

    # Correct the temporal phase
    phi[wrong_pos] += dphi

    # Now the temporal phase is finished, and can be saved 
    # to the result dictionary as well
    result['phi'] = phi

    return result


def get_amplitude(xia):
    """
    Calculates the amplitude.

    Calculates the amplitude of the pressure mode given by the quaternion 
    valued signal xia.

    Parameters
    ----------
    xia : ndarray
        Quaternion valued expression for xi [1] given by 4 real valued time 
        series corresponding to the real, i imaginary, j imaginary and 
        k imaginary parts respectively.
        Shape: (4, M)
    
    Returns
    -------
    amplitude : ndarray
        Amplitude of the flucutuations described by xia.
        Shape: (M,)
    
    References
    ----------
      [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of 
          azimuthal instabilities", Physical Review Fluids, 2018
    """
    return np.sqrt(np.sum(xia**2, axis=0))


def get_chi(xia):
    """
    Calculates the nature angle.

    Calculates the nature angle of the pressure mode given by the quaternion
    valued signal xia.

    Parameters
    ----------
    xia : ndarray
        Quaternion valued expression for xi [1] given by 4 real valued time 
        series corresponding to the real, i imaginary, j imaginary and 
        k imaginary parts respectively.
        Shape: (4, M)
    
    Returns
    -------
    chi : ndarray
        Nature angle of the flucutuations described by xia.
        Shape: (M,)
    
    References
    ----------
      [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of 
          azimuthal instabilities", Physical Review Fluids, 2018
    """
    a, b, c, d = xia / get_amplitude(xia)[None, :]

    return np.arcsin(2 * (b * c - a * d)) / 2


def filter_signal(sig, fs, f_low, f_high, filter_order=5, axis=-1):
    """
    Bandpass filter signal.

    Bandpass filters the incoming signal using a SOS filter.
    
    Parameters
    ----------
    sig : array_like
        Signal(s) to be filtered. The last axis is assumed to be 
        the time axis for the default parameters.
    fs : float
        Sampling frequency in Hertz.
    f_low : float
        Lower frequency of the bandpass filter in Hertz.
    f_high : float
        Higher frequency of the bandpass filter in Hertz.
    filter_order : int, optional
        Order of the bandpass filter. Default: 5
    axis : int, optional
        Axis which the filtering is performed along. Default: -1

    Returns
    -------
    filtered_sig : ndarray
        The filtered signal. Same shape as 'sig'.
    """
    # Translate frequency limits into ratios of the Nyquist frequency
    nyquist = 0.5 * fs
    low = f_low / nyquist
    high = f_high / nyquist
    # Create the filter
    sos = scipy.signal.butter(filter_order, [low, high], analog=False,
                              btype='bandpass', output='sos')
    # Use sosfiltfilt instead of sosfilt to make sure the phase
    # is not shifted due to the single filtering step of sosfilt
    return scipy.signal.sosfiltfilt(sos, sig, axis=axis)
