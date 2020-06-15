from z3 import *

def gen_var2s_subs(z3_vars):
    z3_var2s = {}
    invariant2s = []
    subs = []
    for var in z3_vars:
        if len(z3_vars[var].children()) == 0:
            var2 = str(var) + '2'
            if z3_vars[var].is_int():
                z3_var2s[var2] = z3.Int(var2)
            else:
                z3_var2s[var2] = z3.Real(var2)
            subs.append((z3_vars[var], z3_var2s[var2]))

    return z3_var2s, subs


def gen_rec_constraints(invariants, z3_vars, z3_var2s=None):   # generate the new invariants with the new variables after the loop execution
    invariant2s = []
    if z3_var2s is None:
        z3_var2s, subs = gen_var2s_subs(z3_vars)
    else:
        subs = []
        for var in z3_vars:
            if len(z3_vars[var].children()) == 0:
                var2 = str(var) + '2'
                subs.append((z3_vars[var], z3_var2s[var2]))

    for invariant in invariants:
        invariant2 = z3.substitute(invariant, subs)
        invariant2s.append(invariant2)

    return invariant2s, z3_var2s


def check_invariant(lc, pre, rec, post, invariants, invariants2):
    solver = Solver()

    solver.push()
    solver.add(Not(Implies(pre, And(*invariants))))
    pre_check = solver.check()
    print('precondition proved?', pre_check == z3.unsat)
    if pre_check == z3.sat:
        print(solver.model())
    solver.pop()

    solver.push()
    solver.add(Not(Implies(And(rec, lc, *invariants), And(*invariants2))))
    rec_check = solver.check()
    print('inductive proved?', rec_check == z3.unsat)
    if rec_check == z3.sat:
        print(solver.model())
    solver.pop()

    solver.push()
    solver.add(Not(Implies(And(Not(lc), *invariants), post)))
    post_check = solver.check()
    print('postcondition proved?', post_check == z3.unsat)
    if post_check == z3.sat:
        print(solver.model())
    solver.pop()


def run_check(check, label=''):
    s = Solver()
    s.add(check)
    res = s.check()
    print(label+' correct?', res == unsat)
    if res == sat:
        print('model')
        print(s.model())
        print('check')
        print(s)


