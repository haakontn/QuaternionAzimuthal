# Quaternion structure of azimuthal modes

Implementation of how to obtain the quaternion parameters defined by [Ghirardo and Bothien](https://www.researchgate.net/publication/327755288_Quaternion_structure_of_azimuthal_instabilities "ResearchGate") from pressure time series at discrete points on the annular geometry.


## Usage
Locate the file *quatazim.py* in the folder belonging to the rest of the coding project, and include the following at the top of the main file
```python
from quatazim import quaternion_mode, filter_signal
```
It is expected that the user have measured the azimuthal mode over time at discrete azimuthal locations (`angles`). The pressure values are stored in a 2D array (`pressures`) where the first dimension corresponds to the azimuthal location of the measurement and the second dimension corresponds to the different points in time. The order of the azimuthal mode is determined by the variable `n`.

It is required to band-pass filter the signal around the oscillation frequency of the mode of order `n` as the code is not considering frequency. The included `filter_signal` function can be used for this purpose. Assuming the variables `angles`, `pressures` and `n` are set, the quaternion parameters are obtained by
```python
from quatazim import quaternion_mode, filter_signal

quaternion_parameters = quaternion_mode(angles, pressures, n)
``` 
`quaternion_parameters` is a `OrderedDict` with the following elements
| Key       | Description                                                                   |
|-----------|-------------------------------------------------------------------------------|
| amplitude | Amplitude of the azimuthal mode                                               |
| chi       | Nature angle of the azimuthal mode                                            |
| ntheta_0  | Orientation angle or the azimuthal mode times `n` (location of the anti-node) | 
| phi       | Fast oscillating phase of the azimuthal mode                                  |


See *example.py* for a demonstration on how to use `filter_signal` and `quaternion_mode` on a synthetic time series.

## Notes
- The analytical signal is obtained from the Hilbert transform. Hilbert transforms usually have some end effects, and it can be a good idea to remove the start and end of the returned time series depending on how severe these effects are. See the example in *example.py* for an example of how it might look like. It might also trigger a warning that a few points in the reconstruction is not matching in special circumstances.
- The orientation angles `theta_0` and `theta_0 + np.pi` are completely equivalent. Therefore, it is always possible to add or subtract `np.pi` from this quantity.
- Note there is an upper limit to the mode order based on the number of azimuthal measurement locations. There is currently no check implemented for this, so care should be taken by the user.