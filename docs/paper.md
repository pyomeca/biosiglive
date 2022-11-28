---
title: `Biosiglive`: an Open Sources python package for real-time biosignals processing.
tags:
  - python
  - biomechanics
  - electromyography
  - kinematics
  - dynamics
authors:
  - name: Amedeo Ceglia
    orcid: 
    affiliation: "1"
  - name: Felipe Verdugo
    orcid: 
    affiliation: "x"
  - name: Mickael Begon
    orcid: 0000-0002-4107-9160
    affiliation: "1"
affiliations:
  - name: School of Kinesiology and Exercise Science, Faculty of Medicine, University of Montreal, Canada
    index: 1
date: 
bibliography: paper.bib
---

# Summary
`biosiglive` aims to provide a simple and efficient way to access and process biomechanical data in real-time. 
It was conceived as a user-friendly software aimed at both non-expert and expert programmers. 
The library uses different interfaces to access data from several sources such as motion capture software or any python SDK. 
Interfaces are already implemented for motion capture (Vicon Nexus (Oxford, UK)) and electromyography (EMG) (Delsys trigno utility (Boston, USA)), 
but any interface can be added using the generic class. `biosiglive` was designed for biosignals, 
therefore implemented classes represent data collected from common acquisition systems in biomechanics such as skin markers for motion capture or EMG. 
Methods are available to process in real-time any input signal. Data can be saved in a binary file each time frame to avoid any data loss in case of system shutdown. 
Data can also be displayed using the LivePlot class, which is based on [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) (C++ core) and allows, therefore, fast real-time displaying. 
Finally, 'biosiglive' was conceived as a flexible real-time data processing and streaming tool adaptable to a wide variety of set-ups, software, and systems. 
Therefore, a TCP/IP connection module was implemented to send data to a distant port to be used by any other system. 

# Statement of Need
Biosignals such as EMG or kinematics data are often used to understand human movement in clinical, sports, or even artistic contexts. 
However, the analysis is often time-consuming and requires a good knowledge of programming languages such as MATLAB or Python. 
In the last decade, some open-source and free to use tools have emerged to facilitate the analysis of these signals like `pyomeca` [@martinez2020pyomeca],
`biomechzoo` [@dixon2017biomechzoo], `biomechanical toolkit` [@barre2014biomechanical], or `Kinetics toolkit` [@chenier2021kinetics]. 
`biomechzoo` relies however on Matlab (Mathworks LCC, Naticks, USA), an expensive closed-source software. 
Due to its price, not every programmer in biomechanics can have access to MATLAB. 
The Python environment is on the other hand entirely free and allows anyone to use the package without any cost. 
`pyomeca` and `Kinetics toolkit` both provide efficient methods to analyze biosignals, but are designed to work offline. 
However, real-time use of these signals is often required to provide task feedback to the user [@giggins2013biofeedback] or to control a device [@cozean1988biofeedback]. 
In this type of use, access to biosignals easily and in real-time is a key point. 
To our knowledge, no tool dedicated to biomechanical data is available to provide real-time access and processing of these signals. 
Also, biomechanical data could come from several sources such as a motion capture system or provided python binding SDK. 
So, a package able to stream data in real-time from any of these sources should help the use of biosignals on a larger scale (in clinical, rehabilitation, pedagogical, sport, and artistic activities). 
We have developed `biosiglive` to facilitate the use of biosignals in real time. 
It was achieved by pre-implementing standard data processing and data retrieving from several sources such as Nexus software for motion capture and analogic signals. 
Pre-implemented processing methods are customizable, and the user can develop his/her own method inside the program if needed, 
and add an interface module to make 'biosiglive' work with the desired acquisition system. 
A set of examples is provided to guide the user in using our tool and documentation is available. 

# Features

`biosiglive` is divided in several modules that can be used independently. The mains features are described below.

- `Processing`: real-time processing of the data
- `Interfaces`: interfaces of commons software such as Vicon Nexus or Delsys Trigno Communty.
- `Vizualisation`: real-time signal visualization,
- `Server`: TCP/IP server to disseminate data.
- `Io`: saving data in binary format every time frame.

## A Biomechanical example: Electromyographic pipeline

`biosiglive` provides examples for different biomechanical tasks such as getting and processing EMG signals,
generic analog devices from Nexus, compute live cadence from a treadmill or apply a calibration matrix to raw signals.
More advanced examples are available such as compute and show 3D joint kinematics from a markerset.
The following example shows how to stream, process, display and save EMG signals from Nexus software. 

```python
import numpy as np
import time
from biosiglive import (
LivePlot,
save,
ViconClient,
RealTimeProcessingMethod,
)

# Define the system from which you want to get the data.
vicon_interface = ViconClient(ip="localhost", system_rate=100)
vicon_interface.add_device(nb_channels=1, device_type="emg", name="emg", rate=2000)

# Initialize the plotting method.
emg_plot = LivePlot(name="emg", channel_names=["raw_emg", "processed_emg"], plot_type="curve", nb_subplots=2)
emg_plot.init(use_checkbox=True, plot_windows=vicon_interface.devices[0].rate)

while True:
    # Get the data from the system.
    emg_tmp = vicon_interface.get_device_data()
    # Process the data.
    emg_proc = vicon_interface.devices[0].process(method=RealTimeProcessingMethod.ProcessEmg, moving_average_window=200, normalization=False)
    # Update the plot with the new data.
    data_to_plot = np.concatenate((emg_tmp, emg_proc[:, -1]), axis=0)
    emg_plot.update(data_to_plot)    
    # Save binary file
    save({"raw_emg": emg_tmp, "process_emg":emg_proc[:, -1]}, "emg.bio")

```
The live plot is shown in the following figure.

## TODO : add graph of realtime plot
[Real-time display of raw and processed EMG signals for a one-second window. 
Processing with a moving average downsamples the processed signal by the system rate.
\label{fig:emg_plot}](fig/xxx)


# Research Projects Using `biosiglive`
[@Verdugo2022Feeling]

# Acknowledgements

`biosiglive` is an open-source project created and supported by the Simulation and Movement Modeling (S2M) laboratory located in Montreal.
This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) through the CREATE OPSIDIAN program.

# References
