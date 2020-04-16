import numpy as np
import math as m


def is_same_bank(src, dest, array_per_bank):
    return src/array_per_bank == dest/array_per_bank


def calculate_omega_bw(inputs: np.ndarray):
    size = len(inputs)
    bw = 0
    n_bank = 4
    array_per_bank = size/n_bank
    occupied_local = np.repeat(0, n_bank)
    occupied = np.repeat(0, size)

    for i, target in enumerate(inputs):
        if is_same_bank(i, target, array_per_bank):
            bank_id = int(i // array_per_bank)
            # print(bank_id)
            if not occupied_local[bank_id]:
                occupied_local[bank_id] = 1
                bw += 1
        else:
            if not occupied[target]:
                occupied[target] = 1
                bw += 1
    return bw


def calculate_xbar_dual_bw(inputs: np.ndarray):
    size = len(inputs) / 2
    bw = 0
    n_bank = 4
    array_per_bank = 4
    occupied_local = np.repeat(0, n_bank)
    occupied = np.repeat(0, size)

    for i, target in enumerate(inputs):
        if is_same_bank(i, target, array_per_bank):
            bank_id = int(i // array_per_bank)
            # print(bank_id)
            if not occupied_local[bank_id]:
                occupied_local[bank_id] = 1
                bw += 1
        else:
            target = int(target/2)
            if not occupied[target]:
                occupied[target] = 1
                bw += 1
    return bw


def calculate_omega_bw(inputs: np.ndarray):
    size = len(inputs)
    n_layers = int(m.log(size, 2))
    # print(n_layers)

    valid = n_layers

    def arbitrate(inputs_x, cur_layer):
        invalid = [0] * (n_layers + 1)
        outputs = [invalid, invalid]
        if inputs_x[0][valid]:
            outputs[inputs_x[0][cur_layer]] = inputs_x[0]

            if inputs_x[0][cur_layer] != inputs_x[1][cur_layer]:
                outputs[inputs_x[1][cur_layer]] = inputs_x[1]

        else:
            outputs[inputs_x[1][cur_layer]] = inputs_x[1]

        return outputs

    def gen_next_layer(inputs_x, cur_layer):
        n_switches = int(size / 2)
        outputs = []
        # print(inputs_x)
        for sw in range(0, n_switches):
            up, down = arbitrate([inputs_x[sw + 0], inputs_x[sw + n_switches]], cur_layer)
            outputs.append(up)
            outputs.append(down)
        return outputs

    bits = []
    # print(inputs)
    for input in inputs:
        bins = [int(n) for n in bin(input)[2:].zfill(4)]
        bins.append(1)
        bits.append(bins)

    for layer in range(0, n_layers):
        outputs = gen_next_layer(bits, layer)
        bits = outputs

    return sum([x[valid] for x in bits])


def main():
    loop_count = 100000
    bw = 0
    size = 16
    for i in range(0, loop_count):
        inputs = np.random.randint(0, high=size, size=size)
        # res = calculate_xbar_dual_bw(inputs)
        # res = calculate_xbar_bw(inputs)
        res = calculate_omega_bw(inputs)
        bw += res
        if i % 1000 == 0:
            print(res)
    print(bw/loop_count)


if __name__ == '__main__':
    main()