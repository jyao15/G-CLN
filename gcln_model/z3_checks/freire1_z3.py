from z3 import *
import z3_checks.core

def get_checks(z3_vars, z3_vars2, loop_index=2):
    x, a, r = [z3_vars[v] for v in 'x a r'.split()]
    x2, a2, r2 = [z3_vars2[v] for v in 'x2 a2 r2'.split()]
    assert loop_index == 1
    lc = x > r
    pre = And(a > 0, r == 0, x == ToReal(a)/2)
    rec = And(x2 == x - r, r2 == r + 1, a2 == a)
    post = And(r * r + r >= a, r * r - r < a)
    return lc, pre, rec, post, (), ()
 

def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, a, r = [z3_vars[v] for v in 'x a r'.split()]
    x2, a2, r2 = [z3_vars2[v] for v in 'x2 a2 r2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = x > r
    pre = And(a > 0, r == 0, x == ToReal(a)/2)
    rec = And(x2 == x - r, r2 == r + 1, a2 == a)
    invariant2_with_old_vars = z3.substitute(invariant, [(x, x - r), (r, r + 1)])
    post = And(r * r + r >= a, r * r - r < a)
    solver.add(Not(And(Implies(pre, invariant),
                       # Implies(And(invariant, lc, rec), invariant2),  # z3 timeout
                       Implies(And(invariant, lc), invariant2_with_old_vars),
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
    z3_vars = {'x': Real('x'), 'a': Int('a'), 'r': Int('r')}
    x, a, r = [z3_vars[v] for v in 'x a r'.split()]
    invariant_loop1 = And(a == 2 * x + r * r - r, x >= 0.5)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)


