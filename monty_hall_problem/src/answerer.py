import random


class Answerer:
    def __init__(self, changes: bool) -> None:
        self.changes = changes

    def set(self, n: int) -> None:
        self.n = n

    def select(self) -> int:
        self.x = random.randint(0, self.n - 1)
        return self.x

    def select_final(self, y: int) -> int:
        if not self.changes:
            return self.x
        else:
            cs = []
            for i in range(self.n):
                if i == self.x:
                    pass
                elif i == y:
                    pass
                else:
                    cs.append(i)
        i = random.randint(0, len(cs) - 1)
        return cs[i]
