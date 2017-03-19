import os
import time


targets = [
    'committedInsts::',
    'Slots::0',
    # 'cpi::',
    'ipc::',
    'numCycles',
    'system.cpu.rename.*_Slots',
    'system.cpu.HPTpredIPC::0',
    'HPTQoS',
    # 'system.cpu.*_utilization',
]
