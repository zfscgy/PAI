import sys
import datetime


class Logger:
    def __init__(self, log_stream=sys.stdout, prefix=""):
        self.logger = log_stream
        self.prefix = prefix

    def log(self, info):
        self.logger.write(self.prefix + "[INFO] [" + str(datetime.datetime.today()) + "]  " + info + "\n")

    def logW(self, warning):
        self.logger.write(self.prefix + "[WARN] [" + str(datetime.datetime.today()) + "]  " + warning + "\n")

    def logE(self, error):
        self.logger.write(self.prefix + "[ERROR] [" + str(datetime.datetime.today()) + "]  " + error + "\n")
