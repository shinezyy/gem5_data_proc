with open('./bench_order.txt') as f:
    index_order = [l.strip() for l in f]

full = '_f'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'
suffix = f'{lbuf}{bp}{full}'
