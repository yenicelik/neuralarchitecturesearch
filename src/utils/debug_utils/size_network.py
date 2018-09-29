"""
    This module is copied from
"""

import torch
import gc
import sys
import os
import psutil

PROCESS = psutil.Process(os.getpid())
MEGA = 10 ** 6
MEGA_STR = ' ' * MEGA


#
# def memory_usage_resource():
#     # import resource
#     # rusage_denom = 1024.
#     # if sys.platform == 'darwin':
#     #     # ... it seems that in OSX the output is different units ...
#     #     rusage_denom = rusage_denom * rusage_denom
#     # mem = resource.getrusage(resource.RUSAGE_SELF).ru_isrss / rusage_denom
#     # return mem
#     return os.system("free -m")


def memory_usage_resource():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def memReport():
    return os.system("free -m")

def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)