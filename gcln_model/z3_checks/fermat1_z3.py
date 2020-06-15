from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=2):
    A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
    A2, R2, u2, v2, r2 = [z3_vars2[v] for v in 'A2 R2 u2 v2 r2'.split()]
    if loop_index == 1:
        lc = r != 0
        pre = And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
        rec = And(A2 == A, R2 == R)
        post = And(False)
    elif loop_index == 2:
        lc = r > 0
        pre = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)
        rec = And(r2 == r - v, v2 == v + 2, A2 == A, R2 == R, u2 == u)
        post = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)
    else:
        assert loop_index == 3
        lc = r < 0
        pre = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)
        rec = And(r2 == r + u, u2 == u + 2, A2 == A, R2 == R, v2 == v)
        post = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)
    return lc, pre, rec, post, (), ()
 

def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
    A2, R2, u2, v2, r2 = [z3_vars2[v] for v in 'A2 R2 u2 v2 r2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    if loop_index == 1:
        lc = r != 0
        pre = And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)
        rec = And(A2 == A, R2 == R, u2 == u, v2 == v, r2 == r)
        # post = And(A % ((u - v)/2) == 0)  # z3 timeout
        # post = Exists(temp, A == temp * ((u-v)/2))  # z3 timeout
        post = And(A == ((u + v - 2)/2) * ((u - v)/2))
        pre2_original = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
        pre2_substitute = z3.substitute(pre2_original, subs)
        solver.add(Not(And(Implies(pre, invariant),
                           Implies(And(invariant, lc, rec), pre2_substitute),
                           Implies(And(invariant, Not(lc)), post))))
    elif loop_index == 2:
        lc = r > 0
        pre = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
        rec = And(r2 == r - v, v2 == v + 2, A2 == A, R2 == R, u2 == u)
        post = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
        solver.add(Not(And(Implies(pre, invariant),
                           Implies(And(invariant, lc, rec), invariant2),
                           Implies(And(invariant, Not(lc)), post))))
    else:
        assert loop_index == 3
        lc = r < 0
        pre = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
        rec = And(r2 == r + u, u2 == u + 2, A2 == A, R2 == R, v2 == v)
        post = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
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
    invariant_loop2 = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
    result, model = full_check(z3_vars, invariant_loop2, loop_index=2)
    print(result, model)
    invariant_loop3 = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
    result, model = full_check(z3_vars, invariant_loop3, loop_index=3)
    print(result, model)
