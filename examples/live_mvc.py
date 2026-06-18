"""
This example shows how to use the computeMvc class to calculate the MVC of a online trial set.
You can use an implemented interface and specify it to the computeMvc class by specifying what type of pre-implemented interface you are using.
Once started, the computeMVC class will start in the terminal with questions and you will have to answer them.
Note that if you use this function outside the terminal (pycharm for example) you will have to emulate the terminal in the command window (for pycharm: Run -> Edit Configurations -> Emulate terminal in output console).
First the program will ask you the name of the test you are doing, then the number of seconds you want for the test, it will end at the end (if not specified, it will continue until you press ctrl+c). At the end of the trial, the data will be processed. If you want to change the processing method, you can do so via the set_processing_method before calling the run() method.
You can plot the data (raw, processed or both) at the end of the processing.
Then you can continue with other trials and do the same commands until you decide to stop the program at the end of a trial. In case of malfunction, a temporary file is saved after each trial with all previously recorded data (raw and processed). At the end of all trials, the MVC will be calculated by sorting the data and taking the highest values (for the specified MVC_window). You can then save the data to a file or simply terminate the program which will return a list of MVCs of the length of the number of muscles.
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
