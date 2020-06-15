from z3 import *
import z3_checks.core

def get_checks(z3_vars, z3_vars2, loop_index=2):
    r, p, n, q, h = (z3_vars[v] for v in 'r p n q h'.split())
    r2, p2, n2, q2, h2 = [z3_vars2[v] for v in 'r2 p2 n2 q2 h2'.split()]
    if loop_index == 1:
        lc = q <= n
        pre = And(p == 0, q == 1, r == n, h == 0, n >= 0)
        rec = And(p2 == p, q2 == 4 * q, r2 == r, n2 == n, h2 == h)
        post = And(p == 0, q > n, r == n, h == 0, n >= 0)
    else:
        assert loop_index == 2
        lc = q != 1
        pre = And(p == 0, q > n, r == n, h == 0, n >= 0)
        rec = And(q2 == q / 4, q == 4 * q2, h2 == p + q2, n2 == n,
                  Or(And(r >= h2, p2 == p/2 + q2, p == 2 * p2 - 2 * q2, r2 == r - h2),
                     And(r < h2, p2 == p/2, p == 2 * p2, r2 == r)))
        post = And(p * p <= n, (p + 1) * (p + 1) > n)

    return lc, pre, rec, post, (), ()




def check_valid():
    z3_vars = {str(var):var for var in Ints('p q r h n')}
    # z3_var2s, subs = gen_var2s_subs(z3_vars)
    z3_var2s, subs = core.gen_var2s_subs(z3_vars)


    lc1, pre1, rec1, post1, eqi1, ineqi1 = get_checks(z3_vars, z3_var2s, loop_idx=1)
    lc2, pre2, rec2, post2, eqi2, ineqi2 = get_checks(z3_vars, z3_var2s, loop_idx=2)

    I1 = And(eqi1, ineqi1)
    I1r = substitute(I1, subs)

    I2 = And(eqi2, ineqi2)
    I2r = substitute(I2, subs)

    print('dijkstra loop 1')
    core.check_invariant(lc1, pre1, rec1, post1, [I1], [I1r])

    print('dijkstra loop 2')
    core.check_invariant(lc2, pre2, rec2, post2, [I2], [I2r])


def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    r, p, n, q, h = (z3_vars[v] for v in 'r p n q h'.split())
    r2, p2, n2, q2, h2 = [z3_vars2[v] for v in 'r2 p2 n2 q2 h2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    if loop_index == 1:
        lc = q <= n
        pre = And(p == 0, q == 1, r == n, h == 0, n >= 0)
        rec = And(p2 == p, q2 == 4 * q, r2 == r, n2 == n, h2 == h)
        post = And(p == 0, q > n, r == n, h == 0, n >= 0)
        solver.add(Not(And(Implies(pre, invariant),
                           Implies(And(invariant, lc, rec), invariant2),
                           Implies(And(invariant, Not(lc)), post)
                           )))
    else:
        assert loop_index == 2
        lc = q != 1
        pre = And(p == 0, q > n, r == n, h == 0, n >= 0)
        rec = And(q2 == q / 4, q == 4 * q2, h2 == p + q2, n2 == n,
                  Or(And(r >= h2, p2 == p / 2 + q2, p == 2 * p2 - 2 * q2, r2 == r - h2),
                     And(r < h2, p2 == p / 2, p == 2 * p2, r2 == r)))
        post = And(p * p <= n, (p + 1) * (p + 1) > n)
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
    z3_vars = {v: z3.Int(v) for v in 'r p n q h'.split()}
    r, p, n, q, h = (z3_vars[v] for v in 'r p n q h'.split())
    invariant_loop1 = And(n >= 0, p == 0, h == 0, r == n)
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
    invariant_loop2 = And(r >= 0, r < 2*p + q, p*p + r*q == n*q)
    result, model = full_check(z3_vars, invariant_loop2, loop_index=2)
    print(result, model)
