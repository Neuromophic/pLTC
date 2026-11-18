import logging
import os
from pathlib import Path


def GetMessageLogger(args, setup):
    log_directory = Path(args.logfilepath)
    log_directory.mkdir(parents=True, exist_ok=True)

    LOG_FORMAT = "%(asctime)s - %(levelname)s -\n%(message)s\n"
    formatter = logging.Formatter(LOG_FORMAT)

    log_file = Path(os.path.join(os.fspath(log_directory), f"{setup}.log"))
    is_new_log = not log_file.exists()
    logger_name = f"msglogger_{log_file.resolve()}"
    msglogger = logging.getLogger(logger_name)
    for handler in list(msglogger.handlers):
        handler.close()
        msglogger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    SETUP = vars(args)
    msg = ""
    for k, v in SETUP.items():
        msg = msg + str(k).ljust(20, " ") + str(v) + "\n"

    msglogger.addHandler(file_handler)

    if args.loglevel.lower() == "info":
        msglogger.setLevel(logging.INFO)
    elif args.loglevel.lower() == "debug":
        msglogger.setLevel(logging.DEBUG)

    msglogger.propagate = False
    if is_new_log:
        msglogger.info(msg)
    return msglogger


def PrintParser(args):
    args = vars(args)
    for k, v in args.items():
        print(str(k).ljust(20, " ") + str(v))
