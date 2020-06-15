from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_var2s, loop_idx=2):

    A, B, q, r, d, p = [z3_vars[v] for v in 'A B q r d p'.split()]
    A2, B2, q2, r2, d2, p2 = [z3_var2s[v] for v in 'A2 B2 q2 r2 d2 p2'.split()]
    if loop_idx == 1:
        lc = r >= d
        pre = And(A >= 0, B >= 1, r == A, d == B, p == 1, q == 0)
        rec = And(d2 == 2 * d, p2 == 2 * p, A2 == A, B2 == B, r2 == r, q2 == q)
        post = And(A >= 0, B >= 1, r == A, d == B * p, q == 0, r < d)

    else:
        lc = p != 1
        pre = And(A >= 0, B >= 1, r == A, d == B * p, q == 0, r < d)
        rec = And(A2 == A, B2 == B, d2 == d/2, d == 2 * d2, p2 == p/2, p == 2 * p2, Or(And(r >= d2, r2 == r - d2, q2 == q + p2), And(r < d2, r2 == r, q2 == q)))
        temp = Int('temp')
        post = Exists(temp, And(temp >= 0, temp < B, A == q * B + temp))

    return lc, pre, rec, post, (), ()



def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    A, B, q, r, d, p = [z3_vars[v] for v in 'A B q r d p'.split()]
    A2, B2, q2, r2, d2, p2 = [z3_vars2[v] for v in 'A2 B2 q2 r2 d2 p2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    if loop_index == 1:
        lc = r >= d
        pre = And(A >= 0, B >= 1, r == A, d == B, p == 1, q == 0)
        rec = And(d2 == 2 * d, p2 == 2 * p, A2 == A, B2 == B, r2 == r, q2 == q)
        post = And(A >= 0, B >= 1, r == A, d == B * p, q == 0, r < d)
        solver.add(Not(And(Implies(pre, invariant),
                           Implies(And(invariant, lc, rec), invariant2),
                           Implies(And(invariant, Not(lc)), post))))
    else:
        assert loop_index == 2
        lc = p != 1
        pre = And(A >= 0, B >= 1, r == A, d == B * p, q == 0, r < d)
        rec = And(A2 == A, B2 == B, d2 == d/2, d == 2 * d2, p2 == p/2, p == 2 * p2, Or(And(r >= d2, r2 == r - d2, q2 == q + p2), And(r < d2, r2 == r, q2 == q)))
        temp = Int('temp')
        post = Exists(temp, And(temp >= 0, temp < B, A == q * B + temp))
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
    z3_vars = {v: Int(v) for v in 'A B q r d p'.split()}
    A, B, q, r, d, p = [z3_vars[v] for v in 'A B q r d p'.split()]
    invariant_loop1 = And(A >= 0, B >= 1, q == 0, r == A, d == B*p)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
    invariant_loop2 = And(A == q * B + r, d == B * p, A >= 0, B >= 1, r >= 0, r < d)
    result, model = full_check(z3_vars, invariant_loop2, loop_index=2)
    print(result, model)
