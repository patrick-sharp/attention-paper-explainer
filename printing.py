import os
import traceback
from pathlib import Path


def red(x):
    # ANSI escape characters for red
    return "\033[91m{}\033[00m".format(x)


def green(x):
    # ANSI escape characters for green
    return "\033[92m{}\033[00m".format(x)


def print_clean_exception_traceback(exception):
    """prints the stacktrace and exception text of the exception. Only prints the parts
    of the stacktrace from this project's code (excluding the repl)"""
    tb = exception.__traceback__
    cwd = os.getcwd()
    tb_list = traceback.extract_tb(tb)

    # only show errors that come from this project
    # extremely hacky
    def show_frame_summary(frame_summary):
        cwd = os.getcwd()
        repl_path = Path(cwd, "repl.py")
        is_repl = frame_summary.filename == str(repl_path)
        is_this_project = frame_summary.filename.startswith(cwd)
        return not is_repl and is_this_project

    tb_list = [fs for fs in tb_list if show_frame_summary(fs)]
    for fs in traceback.format_list(tb_list):
        print(fs, end="")
    for line in traceback.format_exception_only(exception):
        print(line, end="")
