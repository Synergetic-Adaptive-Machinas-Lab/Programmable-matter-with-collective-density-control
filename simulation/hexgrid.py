import math

ROWS = 32
COLS = 32

def hex_xy(r, c):

    return (c + (r & 1) * 0.5, r * math.sqrt(3) / 2)

def hex_dist(r1, c1, r2, c2):

    x1, y1 = hex_xy(r1, c1)
    x2, y2 = hex_xy(r2, c2)
    return math.hypot(x1 - x2, y1 - y2)

def cell_key(r, c):
    return r * COLS + c

def key_to_rc(k):
    return k // COLS, k % COLS

def to_user_xy(row, col):

    return col + 1, ROWS - row

def neighbors(r, c, walls=frozenset()):

    dc = (-1, 0) if r % 2 == 0 else (0, 1)
    cands = [
        (r, c - 1), (r, c + 1),
        (r - 1, c + dc[0]), (r - 1, c + dc[1]),
        (r + 1, c + dc[0]), (r + 1, c + dc[1]),
    ]
    return [
        (rr, cc) for rr, cc in cands
        if 0 <= rr < ROWS and 0 <= cc < COLS and cell_key(rr, cc) not in walls
    ]
