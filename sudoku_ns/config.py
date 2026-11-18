"""Rule variant. Default: 'diagonal' (X-Sudoku: both main diagonals also contain 1-9 once).
Override with env SUDOKU_VARIANT=standard (or `main.py --variant standard`)."""
import os

VARIANT = os.environ.get("SUDOKU_VARIANT", "diagonal").lower()
if VARIANT not in ("diagonal", "standard"):
    raise ValueError("SUDOKU_VARIANT must be 'diagonal' or 'standard'")
DIAGONAL = VARIANT == "diagonal"


def units():
    """All constraint units as lists of (row, col): 9 rows, 9 cols, 9 boxes (+2 diagonals)."""
    out = []
    for i in range(9):
        out.append([(i, j) for j in range(9)])
        out.append([(j, i) for j in range(9)])
    for br in range(3):
        for bc in range(3):
            out.append([(3 * br + i, 3 * bc + j) for i in range(3) for j in range(3)])
    if DIAGONAL:
        out.append([(i, i) for i in range(9)])
        out.append([(i, 8 - i) for i in range(9)])
    return out
