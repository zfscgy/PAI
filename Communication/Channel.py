from Communication.Message import ComputationMessage
from Utils.Log import Logger
import sys

class BaseChannel:
    """
    Channel for communication
    """
    def __init__(self, n_clients: int, logger: Logger=None):
        """
        :param n_clients: Number of clients that will Join this channel
        """
        self.n_clients = n_clients
        if logger is None:
            logger = Logger()
        self.logger = logger

    def send(self, receiver: int, msg: ComputationMessage, time_out: float):
        """
        :return: `True` if message is send successfully. If the port is occupied, wait for time_outs.
        """
        raise NotImplementedError()

    def receive(self, sender: int, time_out: float):
        raise NotImplementedError()