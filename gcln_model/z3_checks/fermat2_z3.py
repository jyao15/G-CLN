from z3 import *
import z3_checks.core



def get_checks(z3_vars, z3_vars2, loop_index=2):
    A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
    A2, R2, u2, v2, r2 = [z3_vars2[v] for v in 'A2 R2 u2 v2 r2'.split()]
    assert loop_index == 1
    lc = r != 0
    pre = And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
    rec = And(A2 == A, R2 == R,
              Or(And(r > 0, r2 == r - v, v2 == v + 2, u2 == u),
                 And(r <= 0, r2 == r + u, u2 == u + 2, v2 == v)))
    post = And(A == ((u + v - 2)/2) * ((u - v)/2))
    return lc, pre, rec, post, (), ()
 

def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
    A2, R2, u2, v2, r2 = [z3_vars2[v] for v in 'A2 R2 u2 v2 r2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = r != 0
    pre = And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
    rec = And(A2 == A, R2 == R,
              Or(And(r > 0, r2 == r - v, v2 == v + 2, u2 == u),
                 And(r <= 0, r2 == r + u, u2 == u + 2, v2 == v)))
    post = And(A == ((u + v - 2)/2) * ((u - v)/2))
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
    z3_vars = {v: Int(v) for v in 'A R u v r'.split()}
    A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
    invariant_loop1 = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
