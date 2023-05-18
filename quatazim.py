from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import warnings

import numpy as np
import numpy.typing as npt
import scipy.signal
from numba import njit

@dataclass
class TimeSeries:
    measurement_angles_degree: npt.NDArray[np.float64]
    pressures: npt.NDArray[np.float64]

@dataclass
class Xi:
    xi_1: float | npt.NDArray[np.float64] | npt.NDArray[np.comlex128]
    xi_2: float | npt.NDArray[np.float64] | npt.NDArray[np.comlex128]

    def __truediv__(self, rhs: float | npt.NDArray[np.float64]) -> Xi:
        xi_1 = self.xi_1 / rhs
        xi_2 = self.xi_2 / rhs

        return Xi(xi_1, xi_2)

    @classmethod
    def calculate(cls, time_series: TimeSeries, mode_order: int = 1) -> Xi:
        """
        Calculates the analytical xi values [1].

        Please note the number of measurement locations should be at least one
        higher than the mode number 'n', and the locations has to be well chosen.
        There are no checks to ensure this is satisfied in the code.

        References
        ----------
        [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of 
            azimuthal instabilities", Physical Review Fluids, 2018
        """
        
        # Convert angles to radians and create the least squares matrix
        theta = np.deg2rad(time_series.measurement_angles_degree)
        mat = np.asarray([np.cos(mode_order * theta), np.sin(mode_order * theta)]).T

        # Calculate the two xi values, [xi_1, xi_2]
        xi_1, xi_2 = np.linalg.lstsq(mat, time_series.pressures, rcond=None)[0]

        return cls(xi_1, xi_2)

    def analytical(self) -> Xi:
        """
        Obtain the analytical Xi values.
        """
        xi_1 = Xi.__analytical(self.xi_1)
        xi_2 = Xi.__analytical(self.xi_2)

        return Xi(xi_1, xi_2)

    def as_real_components(self) -> npt.NDArray[np.float64]:
        """
        Get the quaternion valued time series as four real valued
        time series components.

        For a general quaternion number a + bi + cj + dk, the function
        returns a NDArray in the order [a, b, c, d]
        """
        if not isinstance(self.xi_1[0], complex):
            raise ValueError('expected complex valued signals') 

        # To avoid the use of external quaternion libraries, 
        # save the quaternion valued analtyical xi values in
        # a (4, M) shaped array called xia.
        # First index corresponds to the following:
        #   0 = real part
        #   1 = i imaginary part
        #   2 = j imaginary part
        #   4 = k imaginary part

        return np.asarray([self.xi_1.real, self.xi_2.real, self.xi_1.imag, self.xi_2.imag])

    @staticmethod
    def __analytical(signals: npt.NDArray, axis: int = -1) -> npt.NDArray[np.complex128]:
        """
        Compute the analytical signal using the Hilbert transform over axis 'axis'.
        Wrapper for the scipy.signal.hilbert function.
        """
        return scipy.signal.hilbert(signals, axis=axis)


@dataclass
class QuaternionMode:
    """
    Quaternion mode parameters
    """
    amplitude: float | npt.NDArray[np.float64]
    ntheta_0: float | npt.NDArray[np.float64]
    phi: float | npt.NDArray[np.float64]
    chi: float | npt.NDArray[np.float64]


    def asdict(self) -> dict[str, float | npt.NDArray[np.float64]]:
        return dataclasses.asdict(self)

    @staticmethod
    def __amplitude(xia: Xi) -> npt.NDArray[np.float64]: 
        """
        Calculates the amplitude based on the analytical xi values.
        """

        xia_real = xia.as_real_components()
        return np.sqrt(np.sum(xia_real**2, axis=0))

    @classmethod
    def __chi(cls, xia: Xi) -> npt.NDArray[np.float64]:
        """
        Calculates the nature angle based on the analytical xi functions.
        """
        a, b, c, d = (xia / cls.__amplitude(xia)).as_real_components()

        return np.arcsin(2 * (b * c - a * d)) / 2

    @staticmethod
    @njit
    def __unwrap_orientation(ntheta_0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Unwrap the orientation from (-pi/2, pi/2] to span the full (-pi, pi].
        """
        for ind in range(1, len(ntheta_0)):
            diff = ntheta_0[ind] - ntheta_0[ind-1]
            if np.absolute(diff) > np.pi / 2.0:
                # If the difference is above pi/2, move the orientation
                # angle an angle pi to bring it to the correct side of 
                # the Poincare sphere
                theta_tmp = ntheta_0[ind] - np.pi * np.sign(diff)
                # In case the new value is above pi, bring it back into
                # the desired interval again
                ntheta_0[ind] = ((theta_tmp + np.pi) % (2 * np.pi)) - np.pi
            
        return ntheta_0


    @classmethod
    def calculate(cls, time_series: TimeSeries, mode_order: int=1, pi_4_tol: float=1e-8, amp_tol: float=1e-8) -> QuaternionMode:
        """ 
        Calculates the quaternion mode parameters based on pressure time series.
  
        The calculations and parameters are based on a quaternion formalism,
        introduced for annular instabilities by Ghirardo and Bothien [1].
        
        Parameters
        ----------
        time_series : TimeSeries
            Time series of pressure signals, and the corresponding azimuthal
            measurement positions, given as a TimeSeries object.
        mode_order : int
            Azimuthal order of the mode. Default: 1.
        pi_4_tol : float
            Tolerance for how close to +-pi/4 is considered pi/4. Default: 1e-8
        amp_tol : float
            Tolerance for how close the amplitudes have to be in the 
            reconstruction check to be considered fine. If it is above this
            threshold value a warning is triggered. Default: 1e-8

        Returns
        -------
        result : QuaternionMode
            Class containing the time series of the quaternion mode parameters
 
        Notes
        -----
        This function is not conditioning on the frequency at all,
        and to obtain the desired frequency components of the signal the 
        pressure time series should be band pass filtered around the 
        desired frequency. This should also be done if there is a single
        dominant mode on the pressure signals to avoid fitting the
        low frequency noise as well as the desired pressure fluctuations.
        The filter and Hilbert transform might have some end effects, so in 
        both ends of the time series, the reconstruction might not be perfectly 
        representative of the actual pressure time series. Therefore,
        it can be a good idea to discard the first and last part of the
        reconstructed quaternion mode parameters.

        References
        ----------
        [1] G. Ghirardo and M. R. Bothien, "Quaternion structure of
        azimuthal instabilities", Physical Review Fluids, 2018
        """

        if len(time_series.measurement_angles_degree.shape) != 1:
            raise ValueError("expected 'time_series.measurement_angles_degree' to be 1D array")
        if len(time_series.pressures.shape) != 2:
            raise ValueError("expected 'time_series.pressures' to be a 2D array")
        if time_series.measurement_angles_degree.shape[0] != time_series.pressures.shape[0]:
            raise ValueError("first dimension of 'angles' and 'pressures' should be equal")

        # Obtain the analtyical xi functions
        xia = Xi.calculate(time_series, mode_order).analytical()

        # Calculate the parameters, see article Appendix C

        # Find amplitude and nature angle
        amplitude = cls.__amplitude(xia)
        chi = cls.__chi(xia)

        # Normalize the signal for the rest of the calculations
        xia_norm = (xia / cls.__amplitude(xia)).as_real_components()

        # Allocate space for the orientation angle ntheta_0 and the 
        # temporary temporal phase phi_tmp
        ntheta_0 = np.zeros_like(chi)
        phi_tmp = np.zeros_like(chi)

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

        # Unwrap the orientation angle to span (-pi, pi] instead of (-pi/2, pi/2]
        ntheta_0 = cls.__unwrap_orientation(ntheta_0)

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
                       'for some values. Number of differing elements:'
                       f'{nfailed:d}/{mag_check.shape[-1]:d}')
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
        dphi = np.zeros_like(wrong_pos, dtype=np.float64) - np.pi
        # Where the phase was less than 0, the phase correction
        # should be +pi to bring it into the (-pi, pi] interval
        dphi[phi_tmp_wrong < 0] = np.pi

        # Correct the temporal phase
        phi[wrong_pos] += dphi

        return cls(amplitude=amplitude, ntheta_0=ntheta_0, phi=phi, chi=chi)


    def evaluate(self, azimuthal_position: float, mode_order: int=1) -> npt.NDArray[np.float64]:
        """
        Evaluate the value of the mode at a given azimuthal position.
        """
        ndtheta = mode_order * azimuthal_position - self.ntheta_0
        
        cos = np.cos(self.chi) * np.cos(ndtheta) * np.cos(self.phi)
        sin = np.sin(self.chi) * np.sin(ndtheta) * np.sin(self.phi)

        return self.amplitude * (cos + sin)

class Filter:
    """
    Generic filter class used to implement the specific filters.

    Inherit from this class when implementing a new filter type.
    """
    def filtfilt(self, unfiltered_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class Bandpass(Filter):
    """
    Bandpass filter class.

    Used to bandpass filter signals using a SOS-filter.

    Parameters:
    -----------
        sampling_frequency (float): 
            Sampling frequency in Hertz.
        frequency_low (float): 
            Low frequency limit in Hertz.
        frequency_high (float): 
            High frequency limit in Hertz.
        filter_order (int): 
            Order of the filter. Default: 5
        axis (int): 
            Axis to perform the filtering along. Default: -1
    """
    sampling_frequency: float
    frequency_low: float
    frequency_high: float
    filter_order: int = 5
    axis: int = -1

    def filtfilt(self, unfiltered_signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Perform filtering of the given signal.
        """
        # Translate frequency limits into ratios of the Nyquist frequency
        nyquist = 0.5 * self.sampling_frequency
        low = self.frequency_low / nyquist
        high = self.frequency_high / nyquist
        # Create the filter
        sos = scipy.signal.butter(self.filter_order, [low, high], 
                                  analog=False, btype='bandpass', output='sos')
        # Use sosfiltfilt instead of sosfilt to make sure the phase
        # is not shifted due to the single filtering step of sosfilt
        return scipy.signal.sosfiltfilt(sos, unfiltered_signal, axis=self.axis)


################################################################################
#                                Legacy support                                #
################################################################################

def quaternion_mode(angles: npt.NDArray[np.float64], pressures: npt.NDArray[np.float64], 
                    n: int = 1, pi_4_tol: float = 1e-8, amp_tol: float = 1e-8):
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

    time_series = TimeSeries(measurement_angles_degree=angles, pressures=pressures)
    mode = QuaternionMode.calculate(time_series=time_series, mode_order=n, pi_4_tol=pi_4_tol, amp_tol=amp_tol)

    return mode.asdict()
    

def filter_signal(sig: npt.NDArray[np.float64], fs: float, f_low: float, f_high: float, filter_order: int=5, axis: int=-1):
    """
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

    bandpass = Bandpass(sampling_frequency=fs, frequency_low=f_low, frequency_high=f_high, 
                        filter_order=filter_order, axis=axis)
    
    return bandpass.filtfilt(sig)