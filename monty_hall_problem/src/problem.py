import random
from typing import List


class Problem:
    def __init__(self, n: int) -> None:
        self.n = n

    def create(self) -> List[int]:
        p = [0] * self.n
        i = random.randint(0, self.n - 1)
        p[i] = 1
        return p


if __name__ == "__main__":
    p = Problem(5)
    for i in range(10):
        print(p.create())
