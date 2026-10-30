"""Handwritten trusted fixture, not generated program output."""


def matrix_multiplication(a, b):
    rows, inner, columns = len(a), len(b), len(b[0])
    result = [[0 for _ in range(columns)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            value = a[i][k]
            for j in range(columns):
                result[i][j] += value * b[k][j]
    return result
