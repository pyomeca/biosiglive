import asyncio
from biosiglive import load, RealTimeProcessingMethod, InterfaceType, DeviceType, Server, InverseKinematicsMethods
import numpy as np
import time
from biosiglive import QualisysClient


# Fonction pour détecter si Fz dépasse le seuil
def detect_start(previous_f_z, current_f_z, threshold=30):
    # Détection du passage de inférieur à supérieur au seuil
    return previous_f_z <= threshold < current_f_z


class RealTimeDataProcessor:
    def __init__(self, server_ip="192.168.0.1", port=7, data_path="example\\walkAll_LAO01_Cond10.bio",
                 model_path="example\\LAO.bioMod",
                 threshold=30, system_rate=100, device_rate=2000, nb_markers=53, nb_seconds=1):
        # Initialisation du serveur
        self.server = Server(server_ip, port)
        self.server.start()

        # Paramètres
        self.threshold = threshold
        self.system_rate = system_rate
        self.device_rate = device_rate
        self.nb_markers = nb_markers
        self.nb_seconds = nb_seconds

        # Variables d'état
        self.sending_started = False
        self.previous_fz = 0  # Valeur initiale de Fz

        # Chargement des noms des marqueurs
        #self.mks_name = self.load_marker_names()

    def load_marker_names(self):
        # Chargement des noms des marqueurs à partir du fichier
        tmp = load("walkAll_LAO01_Cond10.bio")
        return tmp['markers_names'].data[0:self.nb_markers].tolist()
        return tmp['markers_names'].data[0:self.nb_markers].tolist()

    async def setup_interface(self):
        self.interface = await QualisysClient.create(ip="192.168.254.1", system_rate=100, port=22224)

        # Configuration du jeu de marqueurs

        await self.interface.add_marker_set(
            nb_markers=self.nb_markers,
            data_buffer_size=1000,
            marker_data_file_key="markers",
            name="markers",
            rate=self.system_rate,
            unit="mm"
        )

        await self.interface.add_device(
            nb_channels=18,
            device_type="force_plate",
            name="force_plate",
            data_buffer_size=20000,
            rate=2000,
            device_data_file_key="force_plate",
            processing_method=None,
            moving_average=True,
        )

    async def process_data(self):
        await self.setup_interface()

        while True:
            tic = asyncio.get_event_loop().time()
            packet = await self.interface.Connect.get_current_frame(components=self.interface.component)

            # data recuperation
            mark_tmp = await self.interface.get_marker_set_data(packet=packet)
            dataforce = await self.interface.get_force_plate_data(packet=packet)

            # Calcul de la force verticale moyenne actuelle
            if dataforce is not []:
                current_fz = np.nanmean(dataforce[2])
                current_fz2 = np.nanmean(dataforce[11])
                print(current_fz, current_fz2)
                if not self.sending_started and detect_start(self.previous_fz, current_fz, self.threshold):
                    self.sending_started = True
                    print("Démarrage de l'envoi des données.")

                elif self.sending_started:
                    connection, message = self.server.client_listening()  # Non-bloquant
                    if connection:
                        dataAll = {
                            "Force": dataforce,
                            "Markers": mark_tmp,
                        }
                        self.server.send_data(dataAll, connection, message)

                # Mettre à jour la valeur précédente de Fz
                self.previous_fz = current_fz

            loop_time = time.perf_counter() - tic
            real_time_to_sleep = max(0, (1 / self.system_rate) - loop_time)
            if real_time_to_sleep > 0:
                await asyncio.sleep(real_time_to_sleep)




if __name__ == "__main__":
    processor = RealTimeDataProcessor()
    asyncio.run(processor.process_data())