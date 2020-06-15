from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=2):
    a, b, x, y, z = [z3_vars[v] for v in 'a b x y z'.split()]
    a2, b2, x2, y2, z2 = [z3_vars2[v] for v in 'a2 b2 x2 y2 z2'.split()]
    assert loop_index == 1
    lc = y != 0
    pre = And(a >= 0, b >= 0, x == a, y == b, z == 0)
    rec = And(a2 == a, b2 == b,
              Or(And(y % 2 == 1, z2 == z + x, y2 == (y - 1)/2, x2 == 2 * x),
                 And(y % 2 == 0, z2 == z, y2 == y/2, x2 == 2 * x)))
    post = And(z == a * b)
    return lc, pre, rec, post, (), ()
 
def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    a, b, x, y, z = [z3_vars[v] for v in 'a b x y z'.split()]
    a2, b2, x2, y2, z2 = [z3_vars2[v] for v in 'a2 b2 x2 y2 z2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = y != 0
    pre = And(a >= 0, b >= 0, x == a, y == b, z == 0)
    rec = And(a2 == a, b2 == b,
              Or(And(y % 2 == 1, z2 == z + x, y2 == (y - 1)/2, x2 == 2 * x),
                 And(y % 2 == 0, z2 == z, y2 == y/2, x2 == 2 * x)))
    post = And(z == a * b)
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
    z3_vars = {v: Int(v) for v in 'a b x y z'.split()}
    a, b, x, y, z = [z3_vars[v] for v in 'a b x y z'.split()]
    invariant_loop1 = And(z + x * y == a * b)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
