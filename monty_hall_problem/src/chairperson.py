import random
from typing import List


class Chairperson:
    def __init__(self) -> None:
        pass

    def set(self, p: List[int]) -> None:
        self.answer = p.index(1)
        self.num = len(p)

    def set_number(self, x: int) -> None:
        self.x = x

    def open(self) -> int:
        cs = []
        for i in range(self.num):
            if i == self.x or i == self.answer:
                continue
            cs.append(i)
        m = len(cs)
        k = random.randint(0, m - 1)
        return cs[k]

    def get_answer(self) -> int:
        return self.answer
