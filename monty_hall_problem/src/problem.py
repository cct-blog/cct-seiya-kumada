import random
from typing import List


class Problem:
    def __init__(self) -> None:
        pass

    def create(self) -> List[int]:
        p = [0, 0, 0]
        i = random.randint(0, 2)
        p[i] = 1
        return p


if __name__ == "__main__":
    p = Problem()
    for i in range(10):
        print(p.create())
