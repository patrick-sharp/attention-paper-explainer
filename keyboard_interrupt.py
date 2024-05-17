# Source for this code:
# Gary van der Merwe on stack overflow
# https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py

# This code prevents ctrl-c from interrupting certain code.
# Normally in python you can cancel a function call with ctrl-c.
# If you do that in the middle of saving model state, the mode state will be
# corrupted and un-loadable from disk. This is really annoying if you've been
# training for a while. it's uncommon, but happened to me a couple times during
# development

import signal
import logging


class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


# with DelayedKeyboardInterrupt():
#     # stuff here will not be interrupted by SIGINT
#     critical_code()
