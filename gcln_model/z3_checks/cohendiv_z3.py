from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_var2s, loop_idx=2):

    q, r, a, b, x, y = (z3_vars[v] for v in 'q r a b x y'.split())
    q2, r2, a2, b2, x2, y2 = (z3_var2s[v] for v in 'q2 r2 a2 b2 x2 y2'.split())

    lc = r >= 2*b
    pre = And(x==q*y+r, q>=0, r>=y, x>=1, y>=1, a==1, b==y)
    rec = And(a2==2*a, b2==2*b, q2==q, r2==r, x2==x, y2==y)
    post = And(x==(q+a)*y+(r-b), q+a>=0, r-b>=0, x>=1, y>=1, a>=0, b>=0)

    return lc, pre, rec, post, (), ()


def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    x, y, q, a, b, r = [z3_vars[v] for v in 'x y q a b r'.split()]
    x2, y2, q2, a2, b2, r2 = [z3_vars2[v] for v in 'x2 y2 q2 a2 b2 r2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    if loop_index == 1:
        lc = r >= y
        pre = And(x > 0, y > 0, q == 0, r == x, a == 0, b == 0)
        pre2_original = And(x == q * y + r, r >= 0, x >= 1, y >= 1, a == 1, b == y, r >= y)
        pre2_substitute = z3.substitute(pre2_original, subs)
        rec1 = And(a2 == 1, b2 == y, x2 == x, y2 == y, q2 == q, r2 == r)
        post2_original = And(r >= b, b == y * a, x == q * y + r, r >= 0, x >= 1, y >= 1, r < 2 * b)
        post2_substitute = z3.substitute(post2_original, subs)
        rec2 = And(r2 == r - b, q2 == q + a, x2 == x, y2 == y, a2 == a, b2 == b)
        post = And(x == q * y + r, r >= 0, r < y)
        solver.add(Not(And(Implies(pre, invariant),
                           Implies(And(invariant, lc, rec1), pre2_substitute),
                           Implies(And(post2_substitute, rec2), invariant2),
                           Implies(And(invariant, Not(lc)), post))))
    else:
        assert loop_index == 2
        lc = r >= 2 * b
        pre = And(x == q * y + r, r >= 0, x >= 1, y >= 1, a == 1, b == y, r >= y)
        rec = And(x2 == x, y2 == y, q2 == q, r2 == r, a2 == 2 * a, b2 == 2 * b)
        post = And(r >= b, b == y * a, x == q * y + r, r >= 0, x >= 1, y >= 1, r < 2 * b)
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
    z3_vars = {v: Int(v) for v in 'x y q a b r'.split()}
    x, y, q, a, b, r = [z3_vars[v] for v in 'x y q a b r'.split()]
    invariant_loop1 = And(x == q * y + r, r >= 0, x >= 1, y >= 1)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
    invariant_loop2 = And(r >= b, b == y * a, x == q * y + r, r >= 0, x >= 1, y >= 1)
    result, model = full_check(z3_vars, invariant_loop2, loop_index=2)
    print(result, model)
