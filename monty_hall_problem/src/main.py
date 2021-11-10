import src.answerer as an
import src.chairperson as cp
import src.problem as pr

ITER = 10000
N = 3
if __name__ == "__main__":
    chairperson = cp.Chairperson()
    answerer = an.Answerer(changes=False)
    problem = pr.Problem(N)

    c = 0
    for i in range(ITER):
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
    print(c / ITER)
