"""Small pure-Python input fixture; no benchmark is performed here."""


def matrix_multiplication(a, b):
    """Multiply compatible nonempty rectangular lists of numeric values."""
    rows, inner, columns = len(a), len(b), len(b[0])
    result = [[0 for _ in range(columns)] for _ in range(rows)]
    for i in range(rows):
        for j in range(columns):
            for k in range(inner):
                result[i][j] += a[i][k] * b[k][j]
    return result
