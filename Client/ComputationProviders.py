import threading
import tensorflow as tf
import numpy as np
from Communication.Message import ComputationMessage, MessageType
from Communication.Channel import BaseChannel
from Client.Client import BaseClient
from Utils.Log import Logger
k = tf.keras


class MainTFClient(BaseClient):
    def __init__(self, client_id, channel: BaseChannel, data_clients: list, label_client, logger: Logger=None):
        super(MainTFClient, self).__init__(client_id, channel, logger)
        self.data_clients = data_clients
        self.label_client = label_client

        self.error = False

        #
        self.data_client_outs = dict()

        #
        self.input_tensor = None
        self.network = None
        self.optimizer = None
        self.network_out = None
        self.tape = None

    def build_network(self, network: k.Model, optimizer: k.optimizers):
        self.network = network
        self.optimizer = optimizer

    def __send_config_to(self, client_id: int, config: dict):
        try:
            self.send_check_msg(client_id,  ComputationMessage(MessageType.TRAIN_CONFIG, config))
        except:
            self.logger.logE("Sending configuration to client %d failed." % client_id)
            self.error = True

    def send_config_message(self, config: dict):
        sending_threads = []
        for data_client in self.data_clients + [self.label_client]:
            sending_threads.append(threading.Thread(
                target=self.__send_config_to, args=(data_client, config)))
            sending_threads[-1].start()

        for thread in sending_threads:
            thread.join()
        if self.error:
            return False

    def __send_start_message(self, client_id):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.NEXT_TRAIN_ROUND, None))
        except:
            self.error = True

    def __broadcast_start(self):
        sending_threads = []
        for data_client in self.data_clients + [self.label_client]:
            sending_threads.append(threading.Thread(target=self.__send_start_message, args=(data_client,)))
        for thread in sending_threads:
            thread.join()

    def __recv_client_outs(self, client_id):
        try:
            client_share = self.receive_check_msg(client_id, MessageType.MUL_OUT_SHARE)
            self.data_client_outs[client_share] = client_share
        except:
            self.error = True

    def __gather_client_outs(self):
        gathering_threads = []
        for data_client in self.data_clients:
            gathering_threads.append(threading.Thread(target=self.__recv_client_outs, args=(data_client,)))
            gathering_threads[-1].start()

        for thread in gathering_threads:
            thread.join()
        if self.error:
            self.logger.logE("Exception encountered while receiving client outputs")
            return None

        output_parts = []
        for data_client in self.data_clients:
            output_part = self.data_client_outs[data_client][0]
            for other_client in self.data_clients:
                output_part += self.data_client_outs[data_client][1][other_client] +\
                    self.data_client_outs[other_client][2][data_client]
            output_parts.append(output_part)
        return sum(output_parts)

    def __calculate_output(self, input_np: np.ndarray):
        self.input_tensor = tf.Variable(input_np)
        with tf.GradientTape() as tape:
            self.network_out = self.network(self.input_tensor)
        self.tape = tape

    def __get_output_grad(self):
        self.send_check_msg(self.label_client, ComputationMessage(MessageType.PRED_LABEL, self.network_out.numpy()))
        grad_server_out = self.receive_check_msg(self.label_client, MessageType.PRED_GRAD)
        return grad_server_out

    def __calculate_grad(self, grad_on_output):
        model_jacobians = self.tape.gradient(self.network_out, self.network_out.trainable_parameters())
        model_grad = [tf.reduce_sum(model_jacobian * grad_on_output, axis=[-1, -2]) for model_jacobian in model_jacobians]
        self.optimizer.apply_gradient(zip(model_grad, self.network.trainable_parameters))
        input_jacobian = self.tape.jocabian(self.network_out, self.input_tensor)
        input_grad = tf.reduce_sum(input_jacobian * grad_on_output, axis=[-1, -2]).numpy()
        return input_grad

    def __train_one_batch(self):
        self.__broadcast_start()
        if self.error:
            self.logger.logE("Error encountered while broadcasting start messages")
            return

        client_outputs = self.__gather_client_outs()
        if self.error:
             self.logger.logE("Error encountered while gathering client outputs")
             return

        self.__calculate_output(client_outputs)
        if self.error:
            self.logger.logE("Error encountered while calculating server output")

        try:
            grad_server_out = self.__get_output_grad()
        except:
            self.logger.logE("Error encountered while getting server output gradient")
            return




        self.__calculate_grad()
