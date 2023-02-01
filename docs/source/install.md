# How to install
Biosiglive can be installed from anaconda, pip or from the sources.
## Installing from anaconda
You can install biosiglive from anaconda by running the following command :
```bash
conda install -c conda-forge biosiglive
```
## Installing from pip
You can install biosiglive from pip by running the following command :
```bash
pip install biosiglive
```
pyqtgraph, biorbd and bioviz will not be installed in the same time as they are not available on pip or optional.
So you can install pyqtgraph running the following command :
```bash
pip install pyqtgraph
```
And you can install biorbd and bioviz running the following command :
```bash
conda install -c conda-forge bioviz
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
pip install .
```
