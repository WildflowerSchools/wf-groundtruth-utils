import logging


class Logger(object):
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def logger(self):
        return self._logger


logger = Logger().logger()
