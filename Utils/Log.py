import sys
import datetime
import traceback
import threading


class Logger:
    def __init__(self, log_stream=sys.stdout, prefix="", level=0):
        self.logger = log_stream
        self.prefix = prefix
        self.level = level

        self.write_lock = threading.Lock()

    def log(self, info):
        if 0 >= self.level:
            self.logger.write(self.prefix + "[INFO] [" + str(datetime.datetime.today()) + "]  " + info + "\n")
            self.logger.flush()

    def logW(self, warning):
        if 1 >= self.level:
            self.logger.write(self.prefix + "[WARN] [" + str(datetime.datetime.today()) + "]  " + warning + "\n")
            self.logger.flush()

    def logE(self, error):
        if 2 >= self.level:
            self.write_lock.acquire()
            self.logger.write(self.prefix + "[ERROR] [" + str(datetime.datetime.today()) + "]  " + error + "\n")
            self.logger.write(traceback.format_exc())
            self.logger.write("In thread: " + threading.current_thread().name + "\n")
            self.write_lock.release()
            self.logger.flush()

    def __del__(self):
        if self.logger is not sys.stdout:
            self.logger.close()
