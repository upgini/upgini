# -*- coding: utf-8 -*-
"""
.. module: mdc
.. moduleauthor:: Aljosha Friemann a.friemann@automate.wtf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

from upgini.mdc.context import new_log_context, get_mdc_fields
from pythonjsonlogger import jsonlogger


MDContext = new_log_context
MDC = new_log_context

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.ERROR)


def patch(old_factory):
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)

        for key, value in get_mdc_fields().items():
            setattr(record, key, value)

        return record

    return record_factory


try:
    logging.setLogRecordFactory(patch(logging.getLogRecordFactory()))
except AttributeError:
    logging.LogRecord = patch(logging.LogRecord)


# legacy handler to avoid breaking existing implementations, this will be removed with 2.x
class MDCHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        logging.StreamHandler.__init__(self, *args, **kwargs)
        self.setFormatter(jsonlogger.JsonFormatter())
