"""
Signal handlers for process communication and termination.
"""

import signal
import os
from src import globals as runtime_state

def signal_handler(signum, frame):
    """
    Handler for termination signals (SIGINT, SIGTERM).
    Sets the global termination flag to True.
    """
    if signum in [signal.SIGINT, signal.SIGTERM]:
        print("\n[Main] Termination signal received. Gracefully shutting down...")
        runtime_state.TERMINATE_SIGNAL = True

def request_config_update_handler(signum, frame, shared_data=None):
    """
    Handler for configuration update signals (SIGUSR1).
    Records receipt locally. The shared flag belongs to the coordinator and
    must not be set again by workers receiving its notification.
    
    Args:
        signum: Signal number
        frame: Current stack frame
        shared_data: Accepted for worker callback compatibility
    """
    runtime_state.CONFIG_UPDATE_REQUESTED = True
    print("[Signal] Configuration update requested")

def safe_kill(pid, sig):
    """
    Given a pid, send a signal and handle any exceptions.
    Returns True if the signal was sent successfully, False otherwise.
    """
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        print(f"[SafeKill] PID {pid} not found.")
        return False
    except PermissionError:
        print(f"[SafeKill] PID {pid} cannot send signal.")
        return False
    except Exception as e:
        print(f"[SafeKill] Signal sending error: {e}")
        return False
