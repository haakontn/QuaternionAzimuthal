# Quaternion structure of azimuthal instabilities

Implementation of how to obtain the quaternion parameters defined by [Ghirardo and Bothien](https://www.researchgate.net/publication/327755288_Quaternion_structure_of_azimuthal_instabilities "ResearchGate") from pressure time series at discrete points on the annular geometry.


## Usage
Locate the file *quatazim.py* in the folder belonging to the rest of the coding project, and include the following at the top of the main file
```python
from quatazim import QuaternionMode, TimeSeries, Bandpass
```
It is expected that the user has measured the azimuthal mode over time at discrete azimuthal locations (`measurement_angles_degree`). The pressure values are stored in a 2D array (`pressures`) where the first dimension corresponds to the azimuthal location of the measurement and the second dimension corresponds to the different points in time.

It is required to band-pass filter the signal around the oscillation frequency of the mode of order `mode_order` as the code is not considering frequency. The included `Bandpass` class can be used for this purpose. The quaternion parameters can be obtained from the filtered `pressures` signals as follows
```python
from quatazim import QuaternionMode, TimeSeries

time_series = TimeSeries(measurement_angles_degree, pressures)
quaternion_parameters = QuaternionMode.calculate(time_series, mode_order)
``` 

`quaternion_parameters` is an instance of QuaternionMode, which has the following fields:
| Name      | Description                                                                            |
|-----------|----------------------------------------------------------------------------------------|
| amplitude | Amplitude of the azimuthal mode                                                        |
| ntheta_0  | Orientation angle or the azimuthal mode times `mode_order` (location of the anti-node) | 
| phi       | Fast oscillating phase of the azimuthal mode                                           |
| chi       | Nature angle of the azimuthal mode                                                     |


See *example.py* for a demonstration on how to apply these functions on a synthetic time series.

## Notes
- The analytical signal is obtained from the Hilbert transform. Hilbert transforms usually have some end effects, and it can be a good idea to remove the start and end of the returned time series depending on how severe these effects are. See *example.py* for an example of how it might look like. It might also trigger a warning that a few points in the reconstruction is not matching in certain circumstances.
- The orientation angles `ntheta_0` and `ntheta_0 + np.pi` are completely equivalent. Therefore, it is always possible to add or subtract `np.pi` from this quantity.
- Note there is an upper limit to the mode order based on the number of azimuthal measurement locations. There is currently no check implemented for this, so care should be taken by the user.

## Legacy support

The legacy functions `quaternion_mode(...)` and `filter_signal(...)` still exist for backwards code compatibility.
However, the functions are wrapping the new classes and functionality, which require **Python version 3.7** or newer.

## Installing conda environment

The necessary packages can be installed in a conda environment named `my_environment` with the command
```bash
conda install --name my_environment --file requirements.txt
```

Alternatively, a new environment can be created with
```bash
conda create --name my_new_environment --file requirements.txt
```

## References
**Source of the method:**

[G. Ghirardo and M. R. Bothien, "Quaternion structure of azimuthal instabilities", Physical Review Fluids, 2018](https://www.researchgate.net/publication/327755288_Quaternion_structure_of_azimuthal_instabilities "ResearchGate")

**Papers where this implementation has been used:**

[G. Ghirardo, H. T. Nygård, A. Cuquel, and N. A. Worth "Symmetry breaking modelling for azimuthal combustion dynamics", Proceedings of the Combustion Institute, 2020](https://www.sciencedirect.com/science/article/pii/S1540748920300183 "Elsevier ScienceDirect  (Open access)")

[H. T. Nygård, G. Ghirardo, and N. A. Worth "Azimuthal flame response and symmetry breaking in a forced annular combustor", Combustion and Flame, 2021](https://www.sciencedirect.com/science/article/pii/S0010218021003084 "Elsevier ScienceDirect  (Open access)")