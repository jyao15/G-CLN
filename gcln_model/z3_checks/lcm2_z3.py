from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, y, u, v, a, b = [z3_vars[v] for v in 'x y u v a b'.split()]
    x2, y2, u2, v2, a2, b2 = [z3_vars2[v] for v in 'x2 y2 u2 v2 a2 b2'.split()]
    assert loop_index == 1
    lc = x != y
    pre = And(a >= 1, b >= 1, x == a, y == b, u == b, v == a)
    rec = And(a2 == a, b2 == b,
              Or(And(x > y, x2 == x - y, v2 == v + u, u2 == u, y2 == y),
                 And(x <= y, y2 == y - x, u2 == u + v, v2 == v, x2 == x)))
    post = And(False)

    return lc, pre, rec, post, (), ()
