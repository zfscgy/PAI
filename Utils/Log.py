import sys
import datetime
import traceback

class Logger:
    def __init__(self, log_stream=sys.stdout, prefix="", level=0):
        self.logger = log_stream
        self.prefix = prefix
        self.level = level

    def log(self, info):
        if 0 >= self.level:
            self.logger.write(self.prefix + "[INFO] [" + str(datetime.datetime.today()) + "]  " + info + "\n")

    def logW(self, warning):
        if 1 >= self.level:
            self.logger.write(self.prefix + "[WARN] [" + str(datetime.datetime.today()) + "]  " + warning + "\n")

    def logE(self, error):
        if 2 >= self.level:
            self.logger.write(self.prefix + "[ERROR] [" + str(datetime.datetime.today()) + "]  " + error + "\n")
            self.logger.write(traceback.format_exc() + "\n")