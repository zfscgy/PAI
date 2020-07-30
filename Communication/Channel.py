from Communication.Message import PackedMessage
from Utils.Log import Logger
import sys

class BaseChannel:
    """
    Channel for communication
    """
    def __init__(self, self_id: int, self_port: str, n_clients: int, logger: Logger=None):
        """
        :param n_clients: Number of clients that will Join this channel
        """
        self.client_id = self_id
        self.n_clients = n_clients
        if logger is None:
            logger = Logger()
        self.logger = logger

    def send(self, receiver: int, msg: PackedMessage, time_out: float):
        """
        :return: `True` if message is send successfully. If the port is occupied, wait for time_outs.
        """
        raise NotImplementedError()

    def receive(self, sender: int, time_out: float, key=None, **kwargs) -> PackedMessage:
        raise NotImplementedError()

    def clean(self) -> bool:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()