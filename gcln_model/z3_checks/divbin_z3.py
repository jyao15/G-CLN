from z3 import *
import z3_checks.core
# import core


def get_checks(z3_vars, z3_var2s, loop_idx=2):
    A, B, q, r, b = (z3_vars[v] for v in 'A B q r b'.split())
    A2, B2, q2, r2, b2 = (z3_var2s[v] for v in 'A2 B2 q2 r2 b2'.split())

    if loop_idx == 1:

        lc = r >= b
        pre = And(A > 0, B > 0, q == 0, r == A, b == B)
        rec = And(b2 == 2*b, A2==A, B2==B, q2==q, r2==r)
        post = None  # TODO

        # q = 0, A=r; b > 0; r>0
        eq = And(q == 0, A==r)
        ineq = And(b>0, r>0)

        return lc, pre, rec, post, eq, ineq

    elif loop_idx == 2:

        lc = b != B
        pre = And(q == 0, A == r, b > 0, r > 0, r < b)
        rec = And(A2 == A, B2 == B, b2 == b/2, b == 2 * b2,
                  Or(And(r >= b2, q2 == 2 * q + 1, r2 == r - b2),
                     And(r < b2, q2 == 2 * q, r2 == r)))
        post =  None# TODO


        return lc, pre, rec, post, (), ()


def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    A, B, q, r, b = [z3_vars[v] for v in 'A B q r b'.split()]
    A2, B2, q2, r2, b2 = [z3_vars2[v] for v in 'A2 B2 q2 r2 b2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    if loop_index == 1:
        lc = r >= b
        pre = And(A > 0, B > 0, q == 0, r == A, b == B)
        rec = And(A2 == A, B2 == B, q2 == q, r2 == r, b2 == 2 * b)
        post = And(q == 0, A == r, b > 0, r > 0, r < b)
        solver.add(Not(And(Implies(pre, invariant),
                           Implies(And(invariant, lc, rec), invariant2),
                           Implies(And(invariant, Not(lc)), post))))
    else:
        assert loop_index == 2
        lc = b != B
        pre = And(q == 0, A == r, b > 0, r > 0, r < b)
        rec = And(A2 == A, B2 == B, b2 == b/2, b == 2 * b2,
                  Or(And(r >= b2, q2 == 2 * q + 1, r2 == r - b2),
                     And(r < b2, q2 == 2 * q, r2 == r)))
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
    import core
    z3_vars = {v: Int(v) for v in 'A B q r b'.split()}
    A, B, q, r, b = [z3_vars[v] for v in 'A B q r b'.split()]
    invariant_loop1 = And(q == 0, A == r, b > 0, r > 0)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
    invariant_loop2 = And(b * q - A + r == 0, r < b, r >= 0)
    result, model = full_check(z3_vars, invariant_loop2, loop_index=2)
    print(result, model)
