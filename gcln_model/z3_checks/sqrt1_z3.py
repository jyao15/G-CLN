from z3 import *
import z3_checks.core


def get_checks(z3_vars, z3_vars2, loop_index=1):
    a, n, t, s = (z3_vars[v] for v in 'a n t s'.split())
    a2, n2, t2, s2 = [z3_vars2[v] for v in 'a2 n2 t2 s2'.split()]
    assert loop_index == 1
    lc = s <= n
    pre = And(n >= 0, a == 0, s == 1, t == 1)
    rec = And(n2 == n, a2 == a + 1, t2 == t + 2, s2 == s + t2)
    post = And(a * a <= n, (a + 1) * (a + 1) > n)
    # ref = And(s == (a + 1) * (a + 1), t == 2 * a + 1)
    return lc, pre, rec, post, (), ()
    # rec = And(a2 == a + 1, t2 == t + 2, s2 == s + t + 2, n2 == n)
    # post = And(a*a <= n, (a+1) * (a+1) > n)
    # eqi = And(t == 2 * a + 1, s == (a + 1) * (a + 1))
    # ineqi = And(a * a <= n)



def full_check(z3_vars, invariant, loop_index):
    z3_vars2, subs = core.gen_var2s_subs(z3_vars)
    a, n, t, s = (z3_vars[v] for v in 'a n t s'.split())
    a2, n2, t2, s2 = [z3_vars2[v] for v in 'a2 n2 t2 s2'.split()]
    invariant2 = z3.substitute(invariant, subs)
    solver = z3.Solver()
    assert loop_index == 1
    lc = s <= n
    pre = And(n >= 0, a == 0, s == 1, t == 1)
    rec = And(n2 == n, a2 == a + 1, t2 == t + 2, s2 == s + t2)
    post = And(a * a <= n, (a + 1) * (a + 1) > n)
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
    z3_vars = {v: z3.Int(v) for v in 'a n t s'.split()}
    a, n, t, s = (z3_vars[v] for v in 'a n t s'.split())
    invariant_loop1 = And(a * a <= n, t == 2 * a + 1, s == (a + 1) * (a + 1))
    result, model = full_check(z3_vars, invariant_loop1, loop_index=1)
    print(result, model)
