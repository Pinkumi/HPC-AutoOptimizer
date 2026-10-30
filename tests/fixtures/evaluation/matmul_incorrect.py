"""Handwritten trusted regression fixture reproducing the Qwen indexing mistake."""


def matrix_multiplication(a, b):
    rows, inner, columns = len(a), len(b), len(b[0])
    result = [[0 for _ in range(columns)] for _ in range(rows)]
    for i in range(rows):
        row_a, row_i = a[i], result[i]
        for k in range(inner):
            val_a = row_a[k]
            col_b = [b[r][k] for r in range(rows)]
            for j in range(columns):
                row_i[j] += val_a * col_b[j]
    return result
