import random


class Answerer:
    def __init__(self, changes: bool) -> None:
        self.changes = changes

    def select(self) -> int:
        self.x = random.randint(0, 2)
        return self.x

    def select_final(self, y: int) -> int:
        if not self.changes:
            return self.x
        else:
            for i in range(3):
                if i == self.x:
                    pass
                elif i == y:
                    pass
                else:
                    break
        return i
