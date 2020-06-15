from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, y, c, k = [z3_vars[v] for v in 'x y c k'.split()]
    x2, y2, c2, k2 = [z3_vars2[v] for v in 'x2 y2 c2 k2'.split()]
    assert loop_index == 1
    lc = c < k
    pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
    rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2 * y2 * y2 * y2 * y2)
    post = And(-2 * k * k * k * k * k * k - 6 * k * k * k * k * k -5 * k * k * k * k + k * k + 12 * x == 0)
    return lc, pre, rec, post, (), ()
 

def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, y, c, k = [z3_vars[v] for v in 'x y c k'.split()]
    x2, y2, c2, k2 = [z3_vars2[v] for v in 'x2 y2 c2 k2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = c < k
    pre = And(k >= 0, k <= 30, y == 0, x == 0, c == 0)
    rec = And(c2 == c + 1, k2 == k, y2 == y + 1, x2 == x + y2 * y2 * y2 * y2 * y2)
    post = And(-2 * k * k * k * k * k * k - 6 * k * k * k * k * k -5 * k * k * k * k + k * k + 12 * x == 0)
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
    z3_vars = {v: Int(v) for v in 'x y c k'.split()}
    x, y, c, k = [z3_vars[v] for v in 'x y c k'.split()]
    invariant_loop1 = And(-2 * y * y * y * y * y * y - 6 * y * y * y * y * y -5 * y * y * y * y + y * y + 12 * x == 0, c == y, c <= k)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
