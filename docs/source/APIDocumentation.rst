*****************
API Documentation
*****************

Biosiglive is divided into several modules that can be used combined or separatly. 

Interfaces
==========

Interfaces are use to give a wrapper for python SDK used to retrieve data from a software. There are some pre-implemented interface to retrieve data from Nexus software, Delsys Trigno community or from a biosiglive server. Any custom interface can be created using the `GenericInterface` class an example is available on examples/custom_interface.py.

.. toctree::
   :maxdepth: 2
   
   biosiglive.interfaces

Processing
==========

Some processing functions are available to process biosignal either in real time or offline. Processing examples are provided in the example folder. 

.. toctree::
   :maxdepth: 2
   
   biosiglive.processing
   
File I/O
========

Data can be saved in a pickle binary format using the .bio extension. The particularity is that the data are added to tho file without read it which allow fast data saving. A function is available to read the data saved in the .bio file and return the data as a dictionary. 

.. toctree::
   :maxdepth: 2
   
   biosiglive.file_io
   
GUI
===

The display of the data can be made using the preimplemented functions. It can be a live-updated curve, progress bar, 3D scatter plot or even a 3D skeleton plot (using bioviz library). All these plots, except for the skeleton, are made using the pyqtgraph library which allow fast and modulable display. 

.. toctree::
   :maxdepth: 2
   
   biosiglive.gui
   
Streaming
=========

Here some pipelines are avaible to easily retrieve and disseminate data througth a TCP/IP server. A end-to-end pipeline allowed to compute MVC from EMG data in live during the trials. A class is avaible to create the user custom pipeline to stream data, process them if needed and disseminate them througth a TCP/IP connection, all this using the multiprocessing python library to allowed the live stream. 

.. toctree::
   :maxdepth: 2
   
   biosiglive.streaming


