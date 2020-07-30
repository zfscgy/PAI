from Communication.Message import MessageType, PackedMessage
from Communication.Channel import BaseChannel
from Utils.Log import Logger


class ClientException(Exception):
    def __init__(self, msg):
        super(ClientException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "ClientException:" + self.msg


class BaseClient:
    """
    Base class of client
    """
    def __init__(self, channel: BaseChannel, logger: Logger=None):
        """
        :param client_id: An integer to identify the client
        :type client_id: int
        :param channel: Channel for communication
        """
        self.client_id = channel.client_id
        self.channel = channel
        if not logger:
            logger = Logger()
        self.logger = logger

    def send_msg(self, receiver: int, msg: PackedMessage, time_out=None):
        """
        Send a message to the receiver

        :return: `True` or `False`
        """
        return self.channel.send(receiver, msg, time_out)

    def receive_msg(self, sender: int, time_out=None, key=None, **kwargs):
        return self.channel.receive(sender, time_out, key, **kwargs)

    def receive_check_msg(self, sender: int, header, time_out=None, key=None, **kwargs):
        """
        :param sender:
        :param header: can be MessageType or list of MessageType
        :param time_out:
        :return:
        """
        msg = self.receive_msg(sender, time_out, key, **kwargs)
        if msg is None or (type(header) == MessageType and msg.header != header) or \
                (type(header) == list and msg.header not in header):
            msg = "Expect message type %s, but get message %s" % (str(header), str(msg))
            self.logger.logE(msg)
            raise ClientException("Receive check fails: " + msg)
        return msg

    def send_check_msg(self, receiver: int, msg: PackedMessage, time_out=None):
        resp = self.send_msg(receiver, msg, time_out)
        if resp.header != MessageType.RECEIVED_OK:
            msg = "Send message %s to client %d failed" % (str(msg.header), receiver)
            self.logger.logE(msg)
            raise ClientException("Send check fails: " + msg)
        return True
