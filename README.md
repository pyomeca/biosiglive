# biosiglive
`biosiglive` is a python library dedicated to streaming and processing biosignals in real time from Nexus or Delsys Trigno Community.

# Table of Contents  
[Testing bioptim](#try-bioptim)

[How to install](#how-to-install)
- [From anaconda](#installing-from-anaconda)
- [From the sources](#installing-from-the-sources)

[`biosiglive` API](#biosiglive-api)
- [GUI](#gui)
- [Interfaces](#interfaces)
- [File I/O](#io)
- [Processing](#processing)
- [Streaming](#streaming)

[Examples](#examples)

[Citing](#Citing)

# How to install
Biosiglive can be installed from anaconda or from the sources.
## Installing from anaconda
You can install biosiglive from anaconda by running the following command :
```bash
conda install -c conda-forge biosiglive
```
## Installing from the sources
If installing from the sources, you will need to install the following dependencies from conda (in that particular order):
- [Python](https://www.python.org/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [setuptools](https://pypi.org/project/setuptools/)
- [biorbd](https://github.com/pyomeca/biorbd) (optional: for musculoskeletal models)
- [pyqtgraph](https://www.pyqtgraph.org/) (optional: for real-time plotting)
- [pyopengl](https://www.opengl.org/) (optional: for real-time 3D plotting)
- [bioviz](https://github.com/pyomeca/bioviz) (optional: for skeletal models visualization)

Also, to stream data from Nexus (Vicon) or Trigno Community SDK, you will need to install the following dependencies:
- [pytrigno](https://github.com/aceglia/pytrigno) (optional: for Trigno Community SDK)
- [vicon_dssdk](https://www.vicon.com/software/datastream-sdk/) (optional: for Nexus SDK)

After you can install the package by running the following command in the root folder of the package:
```bash
python setup.py install
```
# `biosiglive` API
## GUI
### LivePlot
The `LivePlot` class allows the plotting of data in a loop. 
Data are updated at each ask of the `update` method. It is possible to ask for a plotting rate so that the update is done at a specific frequency.
Also, it is possible to give to the LivePlot a plot windows, this induces a data buffer that is used to plot the data in a loop.
Available plot types are listed in the enum `PlotType`:
- `PlotType.curve`: plot a curve with a given y data. Colors, titles and labels can be given.
- `PlotType.ProgressBar`: plot a progress bar, the mean value is plotted if several values are given. Name of each bar, units and max values (size of the bar) can be given.
- `PlotType.Scatter3D`: 3D scatter plot. Colors and size of scatters can be given.
- `PlotType.Skeleton`: plot a skeleton. Bioviz keys can be given to plot a specific skeleton.

## Interfaces
### GenericInterface
The `GenericInterface` class is an abstract class that defines the interface to get data from a system. 
It is used to define the methods that should be implemented in a specific interface.
It was used to implement interfaces into `biosiglive` but it can also be used to implement a custom interface (example provided).
The main methods are:
- `add_device`: add a generic device to the interface. The device type should be in the enum `DeviceType` (i.e. "emg", "imu" or "Treadmill").
The created device will inherit from the `Device` class.
- `add_marker_set`: add a marker set to the interface. The marker set should be in the enum `MarkerSetType` (i.e. "labeled").
The created marker set will inherit from the `MarkerSet` class.
- `get_device_data`: get the data of a specific device.
- `get_marker_set_data`: get the data of a specific marker set.

Device and MarkerSet class contain the method to process data in real-time. A buffer is used to store the data.

### ViconClient
The `ViconClient` class is an interface to get data from a Nexus system.
This interface is not available on Linux as the Vicon SDK is not available yet.

### PytrignoClient
The `PytrignoClient` class is an interface to get data from a Trigno Community SDK from Delsys system. 
It allows to stream data from Delsys emg sensors (emg, imu). 
This interface is not available on Linux as the Trigno Community SDK is not available yet.

### TcpClient
The `TcpInterface` class is an interface to get data from a TCP/IP server.

## File I/O
### Save
The `save` function allows to save data in a binary pickle file with the `.bio` extension.
It works by adding data in the file without opening or overwriting it. it's allow to save data in a loop efficiently.

### Load
The `load` function allows to load data from a binary pickle file with the `.bio` extension previously saved using the save function.

## Processing
Methods are provided for live or offline processing of data. 
### LiveProcessing
The `LiveProcessing` class allows to process data in a loop, a data buffer is used for this purpose.
Implemented methods are listed in the enum `RealTimeProcessingMethod`:
- `process_emg`: process EMG data. It is possible to filter the data, to rectify it, to normalize it and to smooth it either with a low-pass filter or a moving average.
- `process_imu`: process IMU data. It is possible to filter the data, to rectify it, to normalize it and to smooth it either with a low-pass filter or a moving average.
- `process_generic_signal`: process generic signal data. It is possible to filter the data, to rectify it, to normalize it and to smooth it either with a low-pass filter or a moving average.
- `calibration_matrix`: apply the calibration matrix to the data.
- `get_peaks`: get the peaks of the data, used in instance to find the cadence from instrumented treadmill.
- `custom`: it is allowed to provide a custom function to process the data. 
The function should take as input at least the new data sample and return the processed data.

### OfflineProcessing
- `process_emg`: process EMG data. It is possible to filter the data, to rectify it, to normalize it and to smooth it either with a low-pass filter or a moving average.
- `process_imu`: process IMU data. It is possible to filter the data, to rectify it, to normalize it and to smooth it either with a low-pass filter or a moving average.
- `process_generic_signal`: process generic signal data. It is possible to filter the data, to rectify it, to normalize it and to smooth it either with a low-pass filter or a moving average.
- `calibration_matrix`: apply the calibration matrix to the data.

### Maximal voluntary contraction trial
In biomechanics, the maximal voluntary contraction (MVC) is often used to compute the maximal value of EMG signals during isometric trials.
The `ComputeMvc` class allows to compute the MVC from a succession of isometric trials using data from any implemented interfaces. Data can be plotted at the end of each trial to check the quality of the data.
At the end of the trials, the maximal values are computed for one non-consecutive second and saved in a pickle file.
a temporary file is created to store the data during the trials to avoid data loss in case of crash.

### Musculoskeletal functions
Musculoskeletal functions are implemented to compute inverse and direct kinematics through `biorbd` package.
- `inverse_kinematics`: compute the inverse kinematics of a given model from markers data. 
Implemented methods are listed in the enum `InverseKinematicsMethods`. A custom method can be provided.
- `direct_kinematics`: return the markers position from given model joint angles.

## Streaming
The `StreamData` class allows to stream data from any interface.
Streamed data can be processed, plotted, saved and even distributed via a TCP/IP server. 
Each task is done in a separate process to avoid blocking the streaming and to allow real-time application.
The performance of the streaming will be affected by the number of computer threads, the number of device, the number of plots or the streaming rate.
An example of streaming is provided in the example folder.

## Examples
A set of example is provided in the `examples` folder.
Every example can be run without any data connection using the custom interface implemented in the example. 
Be aware that if you want to run the example with a real interface, you will need to install the corresponding SDK and to connect to the device.

## How to cite
@misc{Ceglia2022biosiglive,
author = {Ceglia, Amedeo, Felipe, Verdugo and Begon, Mickael},
title = {`Biosiglive`: an Open Sources python package for real-time biosignals processing.},
howpublished={Web page},
url = {https://github.com/pyomeca/biosiglive},
year = {2022}
}










