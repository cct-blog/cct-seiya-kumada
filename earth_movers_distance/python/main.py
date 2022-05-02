import matplotlib.pyplot as plt
PATH = "C:\projects\cct-seiya-kumada\earth_movers_distance\histograms\sea_1.txt"

if __name__ == "__main__":
    ys = []
    s = 0
    for line in open(PATH):
        line = line.strip()
        tokens = line.split()
        s += int(tokens[3])
        ys.append(int(tokens[3]))
    print(s)
    xs = range(len(ys))
    plt.bar(xs, ys)
    plt.xlabel("index")
    plt.ylabel("pixels")
    plt.savefig("./histogram.jpg")
