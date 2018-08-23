#!/usr/bin/env python3
"""
Find free gpus and run stuff from argv on them
Example usage: ./gpustat.py -n 2 env
"""
import os
import sys
from gpustat import new_query

def main():
    cmd = sys.argv[1:]
    n_gpus = 1
    if len(sys.argv) >= 3 and sys.argv[1] == '-n':
        n_gpus = int(sys.argv[2])
        cmd = sys.argv[3:]

    def is_free(gpu):
        return len(gpu.processes) == 0

    gpus = new_query()
    memory_used = [(gpu.memory_used, gpu.index) for gpu in gpus if is_free(gpu)]

    if len(memory_used) < n_gpus:
        print("sorry, there are not enough free gpus right now :(")
        exit(1)

    memory_used.sort()
    indices = [str(idx) for mu, idx in memory_used[:n_gpus]]

    env = os.environ
    env['CUDA_VISIBLE_DEVICES'] = ','.join(indices)

    os.execlpe(cmd[0], *cmd, env)

if __name__ == '__main__':
    main()
