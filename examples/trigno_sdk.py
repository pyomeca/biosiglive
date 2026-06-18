from biosiglive import TrignoSDKClient
from biosiglive import LivePlot, PlotType

if __name__ == "__main__":
    plot_curve = LivePlot(
        name="curve",
        rate=100,
        plot_type=PlotType.Curve,
        nb_subplots=1,
        channel_names=["1"],
    )
    plot_curve.init(plot_windows=10000, y_labels=["emg"])

    client = TrignoSDKClient()
    client.start_streaming()
    while True:
        data, timestamp = client.all_queue["avanti_emg"].get()
        plot_curve.update(data[0:1, :])
