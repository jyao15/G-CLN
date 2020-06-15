from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, y, z, c, k = [z3_vars[v] for v in 'x y z c k'.split()]
    x2, y2, z2, c2, k2 = [z3_vars2[v] for v in 'x2 y2 z2 c2 k2'.split()]
    assert loop_index == 1
    lc = c < k
    pre = And(x == 1, y == z, c == 1, z >= 0, z <= 10, k >= 0, k <= 10)
    rec = And(c2 == c + 1, k2 == k, z2 == z, x2 == x * z + 1, y2 == y * z)
    post = And(x * (z-1) == y - 1)
    return lc, pre, rec, post, (), ()
 
def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, y, z, c, k = [z3_vars[v] for v in 'x y z c k'.split()]
    x2, y2, z2, c2, k2 = [z3_vars2[v] for v in 'x2 y2 z2 c2 k2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = c < k
    pre = And(x == 1, y == z, c == 1, z >= 0, z <= 10, k >= 0, k <= 10)
    rec = And(c2 == c + 1, k2 == k, z2 == z, x2 == x * z + 1, y2 == y * z)
    post = And(x * (z-1) == y - 1)
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
    z3_vars = {v: Int(v) for v in 'x y z c k'.split()}
    x, y, z, c, k = [z3_vars[v] for v in 'x y z c k'.split()]
    invariant_loop1 = And(x * z - x - y + 1 == 0)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
