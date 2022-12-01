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
