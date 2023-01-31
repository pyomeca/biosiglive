******************
About `Biosiglive`
******************

`biosiglive` aims to provide a simple and efficient way to access and process biomechanical data in real time.
It was conceived as user-friendly software aimed for both non-expert and expert programmers.
The library uses interfaces to access data from several sources, such as motion capture software or any Python software development kit (SDK).
Some interfaces are already implemented for Vicon Nexus motion capture (Oxford, UK) and Delsys electromyography SDK (EMG) (Boston, USA). 
That say, any additional interface can be added as custom interface using the abstract class.
`biosiglive` was designed for biosignals, therefore, existing classes represent data collected from standard acquisition systems in biomechanics, 
such as markers for motion capture or EMG. Methods are available to process in real-time any input signal. 
Data can be saved in a binary file at each time frame to avoid any data loss in case of system shutdown. 
Data can also be displayed using the LivePlot class, which is based on (https://github.com/pyqtgraph/pyqtgraph) (C++ core) and allows, 
therefore, fast real-time displaying. 
Finally, 'biosiglive' was conceived as a flexible real-time data processing and streaming tool adaptable to various set-ups, 
software, and systems. 
Therefore, a TCP/IP connection module was implemented to send data to a distant port to be used by any other system.
