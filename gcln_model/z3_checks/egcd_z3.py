from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, y, a, b, p, r, q, s = [z3_vars[v] for v in 'x y a b p r q s'.split()]
    x2, y2, a2, b2, p2, r2, q2, s2 = [z3_vars2[v] for v in 'x2 y2 a2 b2 p2 r2 q2 s2'.split()]
    lc = a != b
    pre = And(x >= 1, y >= 1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)
    rec = And(x2 == x, y2 == y,
              Or(And(a > b, a2 == a - b, p2 == p - q, r2 == r - s, b2 == b, q2 == q, s2 == s),
                 And(a <= b, b2 == b - a, q2 == q - p, s2 == s - r, a2 == a, p2 == p, r2 == r)))
    # this post condition implies a==b==GCD(x,y), z3 does not support GCD so we write it in this way
    post = And(1 == p * s - r * q, a == y * r + x * p, b == x * q + y * s, a == b)
    return lc, pre, rec, post, (), ()
 

def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, y, a, b, p, r, q, s = [z3_vars[v] for v in 'x y a b p r q s'.split()]
    x2, y2, a2, b2, p2, r2, q2, s2 = [z3_vars2[v] for v in 'x2 y2 a2 b2 p2 r2 q2 s2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = a != b
    pre = And(x >= 1, y >= 1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)
    rec = And(x2 == x, y2 == y,
              Or(And(a > b, a2 == a - b, p2 == p - q, r2 == r - s, b2 == b, q2 == q, s2 == s),
                 And(a <= b, b2 == b - a, q2 == q - p, s2 == s - r, a2 == a, p2 == p, r2 == r)))
    # this post condition implies a==b==GCD(x,y), z3 does not support GCD so we write it in this way
    post = And(1 == p * s - r * q, a == y * r + x * p, b == x * q + y * s, a == b)
    solver.add(Not(And(Implies(pre, invariant),
                       Implies(And(invariant, lc, rec), invariant2),
                       Implies(And(invariant, Not(lc)), post))))
    result = solver.check()
    if result == unsat:
        return True, None
    elif result == unknown:
        return False, None
    else:
        assert result == sat
        return False, solver.model()


if __name__ == '__main__':
    z3_vars = {v: Int(v) for v in 'x y a b p r q s'.split()}
    x, y, a, b, p, r, q, s = [z3_vars[v] for v in 'x y a b p r q s'.split()]
    invariant_loop1 = And(1 == p * s - r * q, a == y * r + x * p, b == x * q + y * s)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
