from biosiglive import QualisysClient
from biosiglive import load, RealTimeProcessingMethod, InterfaceType, DeviceType, Server, InverseKinematicsMethods
import numpy as np
import time
import asyncio


# Fonction pour détecter si Fz dépasse le seuil
async def detect_start(previous_f_z, current_f_z, threshold=30):
    # Détection du passage de inférieur à supérieur au seuil
    return previous_f_z <= threshold < current_f_z


class RealTimeDataProcessor:
    def __init__(self, server_ip="192.168.0.1", port=50000,
                 model_path="example\\LAO.bioMod",
                 threshold=30, system_rate=100, device_rate=2000, nb_markers=4, nb_seconds=1):
        # Initialisation du serveur
        self.server = Server(server_ip, port)
        self.server.start()

        # Initialisation de l'interface


        self.model_path = model_path

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

        # Configuration de l'interface
        self.setup_interface()


    async def setup_interface(self):
        # Configuration du jeu de marqueurs
        await self.interface.add_marker_set(
            nb_markers=self.nb_markers,
            data_buffer_size=self.system_rate * self.nb_seconds,
            processing_window=self.system_rate * self.nb_seconds,
            marker_data_file_key="markers",
            name="markers",
            rate=self.system_rate,
            kinematics_method=InverseKinematicsMethods.BiorbdKalman,
            model_path=self.model_path,
            unit="mm",
        )

        # Configuration du dispositif (tapis roulant)
        await self.interface.add_device(
            18,
            name="Treadmill",
            device_type=DeviceType.Generic,
            rate=self.device_rate,
            data_buffer_size=int(self.device_rate * self.nb_seconds),
            processing_window=int(self.device_rate * self.nb_seconds),
            device_data_file_key="treadmill",
        )

    async def process_data(self):
        self.interface = await QualisysClient.create(ip="192.168.254.1", system_rate=100, port=22224)
        queue = asyncio.Queue()
        try:
            while True:
                tic = time.perf_counter()
                packet = await self.interface.Connect.get_current_frame(components=self.interface.component)

                dataforce = await self.interface.get_force_plate_data(packet=packet)
                #Q, _, mark_tmp = self.interface.get_kinematics_from_markers(marker_set_name="markers", get_markers_data=True)

                # data recuperation
                mark_tmp = interface.get_marker_set_data(packet=packet)

                # Calcul de la force verticale moyenne actuelle
                current_fz = np.mean(dataforce[2])

                if not self.sending_started and detect_start(self.previous_fz, current_fz, self.threshold):
                    self.sending_started = True
                    print("Démarrage de l'envoi des données.")

                elif self.sending_started:
                    connection, message = self.server.client_listening()  # Non-bloquant
                    if connection:
                        dataAll = {
                            "Force": dataforce,
                            "Markers": mark_tmp,
                            "MarkersNames": self.mks_name
                        }
                        #"Angle": Q[:, -1],
                        self.server.send_data(dataAll, connection, message)

                # Mettre à jour la valeur précédente de Fz
                self.previous_fz = current_fz
                loop_time = time.perf_counter() - tic
                real_time_to_sleep = max(0, (1/self.system_rate) - loop_time)
                if real_time_to_sleep > 0:
                    await asyncio.sleep(real_time_to_sleep)

        except KeyboardInterrupt:
            print("Arrêt manuel du programme.")
        except Exception as e:
            print(f"Erreur rencontrée : {e}")
        finally:
            #self.server.stop()
            print("Serveur arrêté proprement.")


if __name__ == "__main__":
    processor = RealTimeDataProcessor()
    #processor.process_data()
    asyncio.run(processor.process_data())