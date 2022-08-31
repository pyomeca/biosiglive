# biosiglive
Streaming and processing biosignals in real time from Nexus or Delsys Trigno Community

## Instalation
To install the program run the following comand in the main directory

```bash
python setup.py install
```

## Dependencies

There are some dependencies. You can intall them by running the folowing command :

from PIP
```bash
pip install PyQt5 python-osc
```

From conda forge

```bash
conda install -cconda-forge pyqtgraph 
```
Install pytrigno from this repository : https://github.com/aceglia/pytrigno running the command 
```bash
python setup.py install
```
in the main directory.

## How to use

Biosiglive is a python librairy that allows to share biosignals data from Nexus (Vicon) and trigno community SDK. It includes: live and offline classes for plotting and processing data, classes for streaming data (oppening tcp/ip server and client), classes to efficiently save data with pickle... Some exemples are provided in the examples folder. There are some specific biomechanical application examples to show the possibility of the librairy.

Please feel free to contribute by making pull request or by sending issues.
