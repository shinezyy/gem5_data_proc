with open('./x') as f:
    b = f.readline()
    base = int(b.split('         ')[1])
    b = f.readline()
    wait = int(b.split('         ')[1])
    b = f.readline()
    miss = int(b.split('         ')[1])

    print("calculated performance: ", end=' ')
    print(float(base+miss)/(base+miss+wait))
