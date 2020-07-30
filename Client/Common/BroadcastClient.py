import threading
from Client.Client import BaseClient
from Communication.Message import PackedMessage
from Communication.Channel import BaseChannel
from Utils.Log import Logger


class BroadcastClient(BaseClient):
    def __init__(self, channel: BaseChannel, logger: Logger):
        super(BroadcastClient, self).__init__(channel, logger)
        self.error = False

    def __send_in_thread(self, receiver: int, message: PackedMessage, **kwargs):
        try:
            self.send_check_msg(receiver, message, **kwargs)
        except:
            self.error = True

    def broadcast(self, client_ids: list, message, **kwargs):
        broadcasting_threads = []
        for client_id in client_ids:
            if isinstance(message, dict):
                current_msg = message[client_id]
            else:
                current_msg = message
            broadcasting_threads.append(threading.Thread(target=self.__send_in_thread, args=(client_id, current_msg),
                                                         kwargs=kwargs, name="Broadcast-to-%d" % client_id))
        for broadcasting_thread in broadcasting_threads:
            broadcasting_thread.start()

    def __recv_in_thread(self, sender, header, recv_dict: dict, **kwargs):
        try:
            msg = self.receive_check_msg(sender, header, kwargs)
            recv_dict[sender] = msg.data
        except:
            self.error = True

    def receive_all(self, client_ids: list, header, **kwargs):
        receiving_threads = []
        recv_dict = dict()
        for client_id in client_ids:
            receiving_threads.append(threading.Thread(target=self.__recv_in_thread, args=(client_id, header, recv_dict),
                                                      kwargs=kwargs, name="Receive-from-%d" % client_id))
            receiving_threads[-1].start()
        for receiving_thread in receiving_threads:
            receiving_thread.join()
        return recv_dict
