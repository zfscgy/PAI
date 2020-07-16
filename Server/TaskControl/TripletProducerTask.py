from Server.TaskControl.MPCTask import MPCTask, TaskStage, MPCClientParas, MPCRoles
from Client.MPCProviders.TripletProducer import TripletProducer
from Utils.Log import Logger


class TripletProducerTask(MPCTask):
    def __init__(self, role: str, task_name: str, model_name: str,
                 client_id: int, mpc_paras: MPCClientParas,
                 ip_dict: dict, listen_port: int,
                 configs: dict):
        super(TripletProducerTask, self).__init__(role, task_name, model_name, client_id, mpc_paras,
                                                  ip_dict, listen_port, configs)
        self.stage = TaskStage.Created
        self.crypto_producer = TripletProducer(self.channel, Logger(
            open(self.task_log_path + "triplet_log.txt", "w+"), level=self.log_level
        ), self.mpc_paras)

    def start(self):
        super(TripletProducerTask, self).start()
        self.crypto_producer.start_listening()
