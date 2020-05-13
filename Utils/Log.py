import sys
import datetime


class Logger:
    def __init__(self, log_stream=sys.stdout):
        self.logger = log_stream

    def log(self, info):
        self.logger.write("[INFO] [" + str(datetime.datetime.today()) + "]  " + info + "\n")

    def logW(self, warning):
        self.logger.write("[WARN] [" + str(datetime.datetime.today()) + "]  " + warning + "\n")

    def logE(self, error):
        self.logger.write("[ERROR] [" + str(datetime.datetime.today()) + "]  " + error + "\n")