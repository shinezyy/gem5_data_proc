import numpy as np


def calculate_bw(inputs: np.ndarray):
    size = len(inputs)
    bw = 0
    occupied_local = np.repeat(0, size)
    occupied = np.repeat(0, size)
    for i, target in enumerate(inputs):
        if i == target:
            if not occupied_local[i]:
                occupied_local[i] = 1
                bw += 1
        else:
            if not occupied[target]:
                occupied[target] = 1
                bw += 1
    return bw


def main():
    loop_count = 1000000
    bw = 0
    for i in range(0, loop_count):
        inputs = np.random.randint(0, high=4, size=4)
        res = calculate_bw(inputs)
        bw += res
        if i % 10000 == 0:
            print(res)
    print(bw/loop_count)


if __name__ == '__main__':
    main()