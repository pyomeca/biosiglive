"""
this example shows howto use the computeMvc method class to compute in live the MVC of a set of trials.
You can use your own interface and specify it to the computeMvc class and specify which type of preimplemented
 interface you are using.
When launched the computeMVC class will start in the terminal with question and you will have to answer them.
Note that if you are using this function outside the terminal (pycharm for example) you will have to emulate the
terminal in the command windows (for pycharm -> Run -> Edit Configurations -> Emulate terminal in output console)
First the programm will ask youthe name of the trial you are doing, then the number of second you want for the trial
it will end at the end (if not specified it will continue until a ctrl+c is pressed). At the end of the trial data will
be processed. If you want to change the processing method you can do it via the set_processing_method method before
calling the run() method.
You can plot the data (row, processed or both) at the end of the processing,
Then you can continue to another trials end do the same commands until you decide to stop the program at the end of a
trial. In case of disfunction a temporary file is saved after each trial with all the previous recorded data
(raw and processed). At the end of all trials th MVC will be computed by sorting the data and taking the
MVC_windows higher points. Then you can save the data in a file or just end the program which will return a list of MVC
of the length of the number of muscles.
"""
from biosiglive import ComputeMvc, InterfaceType
from custom_interface import MyInterface

if __name__ == "__main__":
    custom = True
    custom_interface = MyInterface(100, "abd.bio")
    custom_interface.add_device(5, "emg", "my_emg", device_data_file_key="emg")
    interface_type = InterfaceType.Custom if custom else InterfaceType.ViconClient
    list_mvc = ComputeMvc(
        interface_type=interface_type,
        interface_ip="127.0.0.1",
        range_muscle=(0, 5),
        output_file="mvc.bio",
        custom_interface=custom_interface,
    ).run()
    print(list_mvc)
