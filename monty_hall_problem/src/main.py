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

        # 解答者がドアの数（=3）を知る。
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
    size = 4
    problem = pr.Problem(size)

    N = 100
    ITER = 10000

    answerer = an.Answerer(changes=True)
    change_rates = []
    for iter in range(N):
        rate = run(chairperson, answerer, problem, ITER)
        change_rates.append(rate)

    answerer = an.Answerer(changes=False)
    no_change_rates = []
    for iter in range(N):
        rate = run(chairperson, answerer, problem, ITER)
        no_change_rates.append(rate)

    xs = list(range(N))
    plt.scatter(xs, change_rates, label="change", s=1)
    plt.scatter(xs, no_change_rates, label="no change", s=1)
    plt.xlabel("iteration")
    plt.ylabel("probability")
    plt.hlines([1 / size, (size - 1) / size / (size - 2)], xmin=0, xmax=N)
    plt.legend(loc="best")
    plt.savefig("./probability_{}.jpg".format(size))
