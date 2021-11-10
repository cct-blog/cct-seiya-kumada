import matplotlib.pyplot as plt

import src.answerer as an
import src.chairperson as cp
import src.problem as pr


def run(chairperson: cp.Chairperson, answerer: an.Answerer, problem: pr.Problem, iter: int) -> float:
    c = 0
    for i in range(iter):
        # 問題を作る。
        p = problem.create()

        # 司会者が答えを知る。
        chairperson.set(p)

        # 解答者が問題（の数）を知る。
        answerer.set(len(p))

        # 解答者が選択する。
        x = answerer.select()

        # 解答者の番号を記録する。
        chairperson.set_number(x)

        # 解答番号と正解番号以外の扉を開く。
        y = chairperson.open()

        # 解答者が最後の選択をする。
        z = answerer.select_final(y)

        # 答え合わせをする。
        w = chairperson.get_answer()
        if z == w:
            c += 1

    # 正答率
    return c / iter


if __name__ == "__main__":
    chairperson = cp.Chairperson()
    answerer = an.Answerer(changes=True)
    size = 3
    problem = pr.Problem(size)

    start = 100
    end = 20000
    step = 100
    rates = []
    for iter in range(start, end, step):
        rate = run(chairperson, answerer, problem, iter)
        rates.append(rate)

    xs = list(range(start, end, step))
    plt.plot(xs, rates)
    plt.xlabel("iteration")
    plt.ylabel("probability(change)")
    plt.hlines([2 / 3], xmin=0, xmax=end)
    plt.savefig("./change.jpg")
