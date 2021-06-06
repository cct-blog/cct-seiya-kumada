#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import qiskit


# 第2レジスタのビット数
N_EIGEN_STATE = 1


def execute_IQFT_core(circuit, target, n):
    for control in range(0, target):
        circuit.cp(-math.pi / 2 ** (target - control), control, target)
    circuit.h(target)


def execute_IQFT(circuit, n):
    """
    逆フーリエ変換(IQFT)を実行
    """

    # ビットの並びを反転
    for i in range(math.floor(n / 2)):
        circuit.swap(i, n - (i + 1))

    for i in range(n):
        execute_IQFT_core(circuit, i, n)


def make_circuit(n_encode: int):
    """
    量子位相推定を実行する。

    Parameters
    -----
    n_encode: int
        第2レジスタのビット数

    Returns
    -----
    qc: qiskit.QuantumCircuit
        作成した回路
    theta: qiskit.circuit.Parameter
        後で設定する位相
    """

    n = n_encode + N_EIGEN_STATE
    qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(n), qiskit.ClassicalRegister(n_encode))

    # 第2レジスタをUの固有状態にする。
    qc.x(n_encode)

    # 第1レジスターのそれぞれのビットにアダマールゲートを作用させる。
    for qubit in range(n_encode):
        qc.h(qubit)

    # 第1レジスターの各ビットを制御ビットにして第2レジスタにユニタリー操作をおこなる。
    theta = qiskit.circuit.Parameter('φ')
    r = 1
    for c in range(n_encode):
        for i in range(r):
            qc.cp(theta, control_qubit=c, target_qubit=n_encode)
        r *= 2

    qc.barrier()

    # 逆フーリエ変換
    execute_IQFT(qc, n_encode)

    qc.barrier()

    # 第1レジスタの各ビットを測定する。
    for n in range(n_encode):
        # n番目の量子ビットを測定してn番目の古典ビットに保存する。
        qc.measure(n, n)

    return qc, theta
