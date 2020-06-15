from z3 import *
import z3_checks.core

def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, a, r, s = [z3_vars[v] for v in 'x a r s'.split()]
    x2, a2, r2, s2 = [z3_vars2[v] for v in 'x2 a2 r2 s2'.split()]
    assert loop_index == 1
    lc = x > s
    pre = And(a > 0, r == 1, x == ToReal(a), s == 3.25)
    rec = And(x2 == x - s, r2 == r + 1, s2 == s + 6 * r + 3, a2 == a)
    post = And(4 * r * r * r + 6 * r * r + 3 * r >= 4 * a, 4 * a > 4 * r * r * r - 6 * r * r + 3 * r - 1)
    return lc, pre, rec, post, (), ()
 
def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, a, r, s = [z3_vars[v] for v in 'x a r s'.split()]
    x2, a2, r2, s2 = [z3_vars2[v] for v in 'x2 a2 r2 s2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = x > s
    pre = And(a > 0, r == 1, x == ToReal(a), s == 3.25)
    rec = And(x2 == x - s, r2 == r + 1, s2 == s + 6 * r + 3, a2 == a)
    post = And(4 * r * r * r + 6 * r * r + 3 * r >= 4 * a, 4 * a > 4 * r * r * r - 6 * r * r + 3 * r - 1)
    solver.add(Not(And(Implies(pre, invariant),
                       Implies(And(invariant, lc, rec), invariant2),
                       Implies(And(invariant, Not(lc)), post)
                       )))
    result = solver.check()
    if result == unsat:
        return True, None
    elif result == unknown:
        return False, None
    else:
        assert result == sat
        return False, solver.model()


if __name__ == '__main__':
    z3_vars = {'x': Real('x'), 'a': Int('a'), 'r': Int('r'), 's': Real('s')}
    x, a, r, s = [z3_vars[v] for v in 'x a r s'.split()]
    invariant_loop1 = And(4 * r * r * r - 6 * r * r + 3 * r + 4 * x - 4 * a == 1, x > 0, -12 * r * r + 4 * s == 1)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
