from z3 import *
import z3_checks.core


# <<<<<<< HEAD
# def get_checks(z3_vars, z3_vars2, loop_index=2):
    # x1, x2, y1, y2, y3 = (z3_vars[v] for v in 'x1 x2 y1 y2 y3'.split())
    # x12, x22, y12, y22, y32 = [z3_vars2[v] for v in 'x12 x22 y12 y22 y32'.split()]
    # assert loop_index == 1
    # lc = y3 != 0
    # pre = And(x1 >= 0, x2 >= 1, y1 == 0, y2 == 0, y3 == x1)
    # rec = And(x12 == x1, x22 == x2,
              # Or(And(y2 + 1 == x2, y12 == y1 + 1, y22 == 0, y32 == y3 - 1),
                 # And(y2 + 1 != x2, y12 == y1, y22 == y2 + 1, y32 == y3 - 1)))
# =======
def get_checks(z3_vars, z3_var2s, loop_idx=1):
    A, B, q, r, t = (z3_vars[v] for v in 'A B q r t'.split())
    A2, B2, q2, r2, t2 = [z3_var2s[v] for v in 'A2 B2 q2 r2 t2'.split()]
    assert loop_idx == 1
    lc = t != 0
    pre = And(A >= 0, B >= 1, q == 0, r == 0, t == A)
    rec = And(A2 == A, B2 == B,
              Or(And(r + 1 == B, q2 == q + 1, r2 == 0, t2 == t - 1),
                 And(r + 1 != B, q2 == q, r2 == r + 1, t2 == t - 1)))
    temp = Int('temp')
    post = Exists(temp, And(A == q * B + temp, temp >= 0, temp < B))
    return lc, pre, rec, post, (), ()


def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    A, B, q, r, t = (z3_vars[v] for v in 'A B q r t'.split())
    A2, B2, q2, r2, t2 = [z3_vars2[v] for v in 'A2 B2 q2 r2 t2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = t != 0
    pre = And(A >= 0, B >= 1, q == 0, r == 0, t == A)
    rec = And(A2 == A, B2 == B,
              Or(And(r + 1 == B, q2 == q + 1, r2 == 0, t2 == t - 1),
                 And(r + 1 != B, q2 == q, r2 == r + 1, t2 == t - 1)))
    temp = Int('temp')
    post = Exists(temp, And(A == q * B + temp, temp >= 0, temp < B))
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
    z3_vars = {v: z3.Int(v) for v in 'A B q r t'.split()}
    A, B, q, r, t = (z3_vars[v] for v in 'A B q r t'.split())
    invariant_loop1 = And(q * B + r + t == A, r < B, r >= 0)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
