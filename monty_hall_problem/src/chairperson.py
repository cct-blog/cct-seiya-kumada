import random
from typing import List


class Chairperson:
    def __init__(self) -> None:
        pass

    def set(self, p: List[int]) -> None:
        self.problem = p
        for i in range(3):
            if self.problem[i] == 1:
                self.answer = i
                break

    def set_answer(self, x: int) -> None:
        self.x = x

    def open(self) -> int:
        cs = []
        for i in range(3):
            if i == self.x or i == self.answer:
                continue
            cs.append(i)
        if 2 == len(cs):
            k = random.randint(0, 1)
            return cs[k]
        else:
            return cs[0]

    def get_answer(self) -> int:
        return self.answer
