import numpy as np
import threading
import time
from Client.MPCClient import MPCClient, MPCClientParas, ClientMode
from Communication.Message import MessageType, PackedMessage
from Communication.Channel import BaseChannel
from Utils.Log import Logger


class TripletProducer(MPCClient):
    """
    A client for generate beaver triples. It can listen to other clients
    """
    def __init__(self, channel: BaseChannel, logger:Logger, mpc_paras: MPCClientParas, listen_clients):
        super(TripletProducer, self).__init__(channel, logger, mpc_paras)
        self.listen_client_ids = listen_clients
        self.triplet_proposals = dict()
        self.listening_thread = [None for _ in range(channel.n_clients)]
        self.listening = False

        self.last_msg_time = time.time()

    def __listen_to(self, sender: int):
        msg = self.receive_msg(sender, time_out=120)
        if msg is not None:
            if msg.header == MessageType.Triplet_Set:
                operand_sender, target, shape_sender, shape_target = msg.data
                existing_shapes = self.triplet_proposals.get((target, sender))
                if existing_shapes is not None:
                    if existing_shapes[1] == shape_target and existing_shapes[2] == shape_sender:
                        self.__generate_and_send_triplets(existing_shapes[0], (target, sender), (shape_target, shape_sender))
                        del self.triplet_proposals[(target, sender)]
                    else:
                        self.logger.logW("Triplet shapes %s %s not match with clients %d and %d" % (shape_target, shape_sender, target, sender))
                else:
                    self.triplet_proposals[(sender, target)] = (operand_sender, shape_sender, shape_target)
            else:
                self.logger.logW("Expect SET_TRIPLET message, but received %s from %d" % (msg.header, sender))
            return True
        else:
            return False

    def __receive_stop_msg(self):
        # Check if there's main client's training stop message every 15 seconds.
        while True:
            try:
                self.receive_check_msg(self.main_client_id, MessageType.Common_Stop, key="Stop", time_out=15, interval=5)
            except Exception as e:
                if not self.listening:
                    self.stop_listening()
                    return False
                continue
            break
        self.logger.log("Received stop message from main_client. Stop listening. ")
        self.stop_listening()
        return True

    def __generate_and_send_triplets(self, op_position: int, clients, shapes):
        # 判断哪一个是乘数，哪一个是被乘数
        if op_position == 2:
            shapes = [shapes[1], shapes[0]]
            clients = [clients[1], clients[0]]
        u0 = np.random.uniform(-1, 1, shapes[0])
        u1 = np.random.uniform(-1, 1, shapes[0])
        v0 = np.random.uniform(-1, 1, shapes[1])
        v1 = np.random.uniform(-1, 1, shapes[1])
        z = np.matmul(u0 + u1, v0 + v1)
        z0 = z * np.random.uniform(0, 1, z.shape)
        z1 = z - z0
        try:
            self.send_check_msg(clients[0], PackedMessage(MessageType.Triplet_Array, (clients[1], u0, v0, z0), clients[1]))
        except:
            self.logger.logE("Sending triplets to client %d failed" % clients[0])
        try:
            self.send_check_msg(clients[1], PackedMessage(MessageType.Triplet_Array, (clients[0], v1, u1, z1), clients[0]))
        except:
            self.logger.logE("Sending triplets to client %d failed" % clients[1])

    def __listen_to_client(self, sender_id):
        while self.listening:
            if self.__listen_to(sender_id):
                self.last_msg_time = time.time()
            else:
                if time.time() - self.last_msg_time > 120:
                    if self.listening:
                        self.logger.logW("Did not receive any message for 120s. Stop listening.")
                        self.listening = False

    def start_listening(self):
        """
        Start the listening thread
        """
        self.listening = True
        for i in self.listen_client_ids:
            self.listening_thread[i] = threading.Thread(target=self.__listen_to_client, args=(i,),
                                                        name="Triplet-Listening-to-%d" % i)
            self.listening_thread[i].start()
        self.logger.log("Started listening to clients: {}".format(self.listen_client_ids))
        return self.__receive_stop_msg()

    def stop_listening(self):
        """
        Stop the listening thread
        """
        self.listening = False
        for i in self.listen_client_ids:
            self.listening_thread[i].join()
