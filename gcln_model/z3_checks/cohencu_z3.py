from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_var2s, loop_idx=2):

    a, n, x, y, z = [z3_vars[v] for v in 'a n x y z'.split()]
    a2, n2, x2, y2, z2 = [z3_var2s[v] for v in 'a2 n2 x2 y2 z2'.split()]
    lc = n <= a
    pre = And(n == 0, x == 0, y == 1, z == 6, a >= 0)
    rec = And(n2 == n + 1, x2 == x + y, y2 == y + z, z2 == z + 6, a2 == a)
    post = And(x == (a + 1) * (a + 1) * (a + 1))

    return lc, pre, rec, post, (), ()

def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    a, n, x, y, z = [z3_vars[v] for v in 'a n x y z'.split()]
    a2, n2, x2, y2, z2 = [z3_vars2[v] for v in 'a2 n2 x2 y2 z2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = n <= a
    pre = And(n == 0, x == 0, y == 1, z == 6, a >= 0)
    rec = And(n2 == n + 1, x2 == x + y, y2 == y + z, z2 == z + 6, a2 == a)
    post = And(x == (a + 1) * (a + 1) * (a + 1))
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
    z3_vars = {v: Int(v) for v in 'a n x y z'.split()}
    a, n, x, y, z = [z3_vars[v] for v in 'a n x y z'.split()]
    invariant_loop1 = And(z == 6 * n + 6, y == 3 * n * n + 3 * n + 1, x == n * n * n, n <= a + 1)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
