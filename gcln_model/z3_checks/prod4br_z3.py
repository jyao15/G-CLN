from z3 import *
import z3_checks.core

def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, y, a, b, p, q = [z3_vars[v] for v in 'x y a b p q'.split()]
    x2, y2, a2, b2, p2, q2 = [z3_vars2[v] for v in 'x2 y2 a2 b2 p2 q2'.split()]
    assert loop_index == 1
    lc = And(a != 0, b != 0)
    pre = And(x >= 1, y >= 1, x == a, y == b, p == 1, q == 0)
    rec = And(x2 == x, y2 == y,
              Or(And(a % 2 == 0, b % 2 == 0, a2 == a/2, b2 == b/2, p2 == 4 * p, q2 == q),
                 And(a % 2 == 1, b % 2 == 0, a2 == a - 1, b2 == b, p2 == p, q2 == q + b * p),
                 And(a % 2 == 0, b % 2 == 1, a2 == a, b2 == b - 1, p2 == p, q2 == q + a * p),
                 And(a % 2 == 1, b % 2 == 1, a2 == a - 1, b2 == b - 1, p2 == p, q2 == q + (a2 + b2 + 1) * p)))
    post = And(q == x * y)
    return lc, pre, rec, post, (), ()
 
def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, y, a, b, p, q = [z3_vars[v] for v in 'x y a b p q'.split()]
    x2, y2, a2, b2, p2, q2 = [z3_vars2[v] for v in 'x2 y2 a2 b2 p2 q2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = And(a != 0, b != 0)
    pre = And(x >= 1, y >= 1, x == a, y == b, p == 1, q == 0)
    rec = And(x2 == x, y2 == y,
              Or(And(a % 2 == 0, b % 2 == 0, a2 == a/2, b2 == b/2, p2 == 4 * p, q2 == q),
                 And(a % 2 == 1, b % 2 == 0, a2 == a - 1, b2 == b, p2 == p, q2 == q + b * p),
                 And(a % 2 == 0, b % 2 == 1, a2 == a, b2 == b - 1, p2 == p, q2 == q + a * p),
                 And(a % 2 == 1, b % 2 == 1, a2 == a - 1, b2 == b - 1, p2 == p, q2 == q + (a2 + b2 + 1) * p)))
    post = And(q == x * y)
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
    z3_vars = {v: Int(v) for v in 'x y a b p q'.split()}
    x, y, a, b, p, q = [z3_vars[v] for v in 'x y a b p q'.split()]
    invariant_loop1 = And(q + a * b * p == x * y)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
