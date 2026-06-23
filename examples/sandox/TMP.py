
#for d, device in enumerate(self.devices):
for platenum in range(len(forcesdata)):
    PFForce = forcesdata[platenum][1]
    device.new_data = np.zeros((9, len(PFForce)))
    data_tmp = np.array(PFForce)
    data_tmp = data_tmp.T
    count = 0
    for output_name, channel_name, unit in device.infos:
        device.new_data[count, :] = data_tmp[count]
        device.channel_names.append(channel_name)
        count += 1
        if count == device.nb_channels:
            break
    device_data = device_data if channel_idx else device.new_data
    device.append_data(device.new_data)
    all_device_data.append(device_data)
if len(all_device_data) == 1:
    return all_device_data[0]
return all_device_data