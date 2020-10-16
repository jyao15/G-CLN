from z3 import *
import z3_checks

def check_nla_invariant(invariant, name, loop_idx=1, ctx_simplify=False):

    doc_invariant = And(False)


    # if problem == 'ps2':
        # x, y, c, k = Ints('x y c k')
        # ground_truth = And(2*x - y*y -y == 0, c == y, c<=k)

    # def full_z3_validation(name, loop_idx, z3_vars, invariant):
    if name == 'cohencu':
        z3_vars = {v: Int(v) for v in 'a n x y z'.split()}
        a, n, x, y, z = [z3_vars[v] for v in 'a n x y z'.split()]
        invariant_loop1 = And(z == 6 * n + 6, y == 3 * n * n + 3 * n + 1, x == n * n * n, n <= a + 1)
        doc_invariant = invariant_loop1


    if name == 'cohendiv':
        z3_vars = {v: Int(v) for v in 'x y q a b r'.split()}
        x, y, q, a, b, r = [z3_vars[v] for v in 'x y q a b r'.split()]
        invariant_loop1 = And(x == q * y + r, r >= 0, x >= 1, y >= 1)
        invariant_loop2 = And(r >= b, b == y * a, x == q * y + r, r >= 0, x >= 1, y >= 1)

        doc_invariant = invariant_loop1
        if loop_idx == 2:
            doc_invariant = invariant_loop2

    if name == 'dijkstra':
        z3_vars = {v: z3.Int(v) for v in 'r p n q h'.split()}
        r, p, n, q, h = (z3_vars[v] for v in 'r p n q h'.split())
        invariant_loop1 = And(n >= 0, p == 0, h == 0, r == n)
        invariant_loop2 = And(r >= 0, r < 2*p + q, p*p + r*q == n*q)

        doc_invariant = invariant_loop1
        if loop_idx == 2:
            doc_invariant = invariant_loop2

    if name == 'divbin':
        z3_vars = {v: Int(v) for v in 'A B q r b'.split()}
        A, B, q, r, b = [z3_vars[v] for v in 'A B q r b'.split()]
        invariant_loop1 = And(q == 0, A == r, b > 0, r > 0)
        invariant_loop2 = And(b * q - A + r == 0, r < b, r >= 0)

        doc_invariant = invariant_loop1
        if loop_idx == 2:
            doc_invariant = invariant_loop2

    if name == 'egcd':
        z3_vars = {v: Int(v) for v in 'x y a b p r q s'.split()}
        x, y, a, b, p, r, q, s = [z3_vars[v] for v in 'x y a b p r q s'.split()]
        invariant_loop1 = And(1 == p * s - r * q, a == y * r + x * p, b == x * q + y * s)
        doc_invariant = invariant_loop1

    if name == 'egcd2':
        z3_vars = {v: Int(v) for v in 'x y a b p r q s c k GCD(x,y) GCD(a,b)'.split()}
        x, y, a, b, p, r, q, s, c, k, gcdxy, gcdab= [z3_vars[v] for v in 'x y a b p r q s c k GCD(x,y) GCD(a,b)'.split()]
        invariant_loop1 = And(q*x + s*y == b, p*x + r*y == a, gcdxy == gcdab)
        invariant_loop2 = And(a == k*b+c, gcdab == gcdxy, a==y*r+x*p, b==x*q+y*s )
        
        doc_invariant = invariant_loop1
        if loop_idx == 2: 
            invariant = And ( And(invariant) , invariant_loop1)
            doc_invariant = invariant_loop2 

    if name == 'egcd3':
        z3_vars = {v: Int(v) for v in 'x y a b p r q s c k d v  GCD(x,y) GCD(a,b)'.split()}
        x, y, a, b, p, r, q, s, c, k, d, v, gcdxy, gcdab= [z3_vars[v] for v in 'x y a b p r q s c k d v  GCD(x,y) GCD(a,b)'.split()]
        invariant_loop1 = And(q*x + s*y == b, p*x + r*y == a, gcdxy == gcdab)
        invariant_loop2 = And(a == k*b+c, gcdab == gcdxy, a==y*r+x*p, b==x*q+y*s )
        invariant_loop3 = And(a == k*b+c, gcdab == gcdxy, a==y*r+x*p, b==x*q+y*s,v == b*d )
        
        doc_invariant = invariant_loop1
        if loop_idx == 2: 
            invariant = And ( And(invariant) , invariant_loop1)
            doc_invariant = invariant_loop2
        if loop_idx == 3: 
            invariant = And ( And(invariant) , invariant_loop1, invariant_loop2)
            doc_invariant = invariant_loop3

    if name == 'fermat1':
       z3_vars = {v: Int(v) for v in 'A R u v r'.split()}
       A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
       invariant_loop1 = And(4 * A + 4*r == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
       invariant_loop2 = And(4 * A + 4*r == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)
       invariant_loop3 = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 3)
       doc_invariant = invariant_loop1
       if loop_idx == 2:
           doc_invariant = invariant_loop2
       if loop_idx == 3:
           doc_invariant = invariant_loop3

    if name == 'fermat2':
        z3_vars = {v: Int(v) for v in 'A R u v r'.split()}
        A, R, u, v, r = [z3_vars[v] for v in 'A R u v r'.split()]
        invariant_loop1 = And(4 * (A + r) == u * u - v * v - 2 * u + 2 * v, v % 2 == 1, u % 2 == 1, A >= 1)
        doc_invariant = invariant_loop1

    if name == 'freire1':
        # z3_vars = {'x': Real('x'), 'a': Int('a'), 'r': Int('r')}
        z3_vars = {'x': Int('x'), 'a': Int('a'), 'r': Int('r')}
        x, a, r = [z3_vars[v] for v in 'x a r'.split()]
        invariant_loop1 = And(a == 2 * x + r * r - r, x >= 0)
        doc_invariant = invariant_loop1

    if name == 'freire2':
        # z3_vars = {'x': Real('x'), 'a': Int('a'), 'r': Int('r'), 's': Real('s')}
        z3_vars = {'x': Int('x'), 'a': Int('a'), 'r': Int('r'), 's': Int('s')}
        x, a, r, s = [z3_vars[v] for v in 'x a r s'.split()]
        invariant_loop1 = And(4 * r * r * r - 6 * r * r + 3 * r + 4 * x - 4 * a == 1, x >= 0, -12 * r * r + 4 * s == 1)
        doc_invariant = invariant_loop1

    if name == 'geo1':
        z3_vars = {v: Int(v) for v in 'x y z c k'.split()}
        x, y, z, c, k = [z3_vars[v] for v in 'x y z c k'.split()]
        invariant_loop1 = And(x * z - x - y + 1 == 0)
        doc_invariant = invariant_loop1

    if name == 'geo2':
        z3_vars = {v: Int(v) for v in 'x y z c k'.split()}
        x, y, z, c, k = [z3_vars[v] for v in 'x y z c k'.split()]
        invariant_loop1 = And(x * z - x - z * y + 1 == 0)
        doc_invariant = invariant_loop1

    if name == 'geo3':
        z3_vars = {v: Int(v) for v in 'x y z c k a'.split()}
        x, y, z, c, k, a = [z3_vars[v] for v in 'x y z c k a'.split()]
        invariant_loop1 = And(x * z - x + a - a * z * y == 0)
        doc_invariant = invariant_loop1

    if name == 'hard':
        z3_vars = {v: Int(v) for v in 'A B q r d p'.split()}
        A, B, q, r, d, p = [z3_vars[v] for v in 'A B q r d p'.split()]
        invariant_loop1 = And(A >= 0, B >= 1, q == 0, r == A, d == B*p)
        invariant_loop2 = And(A == q * B + r, d == B * p, A >= 0, B >= 1, r >= 0, r < d)

        doc_invariant = invariant_loop1
        if loop_idx == 2:
            doc_invariant = invariant_loop2

    if name == 'lcm1':
        z3_vars = {v: Int(v) for v in ('a', 'b', 'x', 'y', 'u', 'v', 'GCD(x,y)', 'GCD(a,b)')}
        a, b, x, y, u, v, gcdxy, gcdab = (z3_vars[v] for v in ('a', 'b', 'x', 'y', 'u', 'v', 'GCD(x,y)', 'GCD(a,b)'))
        doc_invariant = invariant_loop1 = invariant_loop2 = invariant_loop3 = And(gcdab == gcdxy, x * u + y * v == a * b)

    if name == 'lcm2':
        z3_vars = {v: Int(v) for v in ('a', 'b', 'x', 'y', 'u', 'v', 'GCD(x,y)', 'GCD(a,b)')}
        a, b, x, y, u, v, gcdxy, gcdab = (z3_vars[v] for v in ('a', 'b', 'x', 'y', 'u', 'v', 'GCD(x,y)', 'GCD(a,b)'))
        invariant_loop1 = And(gcdab == gcdxy, x * u + y * v == 2 * a * b, x > 0, y > 0)
        doc_invariant = invariant_loop1

    if name == 'mannadiv':
        z3_vars = {v: z3.Int(v) for v in 'A B q r t'.split()}
        A, B, q, r, t = (z3_vars[v] for v in 'A B q r t'.split())
        invariant_loop1 = And(q * B + r + t == A, r < B, r >= 0)
        doc_invariant = invariant_loop1

    if name == 'prod4br':
        z3_vars = {v: Int(v) for v in 'x y a b p q'.split()}
        x, y, a, b, p, q = [z3_vars[v] for v in 'x y a b p q'.split()]
        invariant_loop1 = And(q + a * b * p == x * y)
        doc_invariant = invariant_loop1

    if name == 'prodbin':
        z3_vars = {v: Int(v) for v in 'a b x y z'.split()}
        a, b, x, y, z = [z3_vars[v] for v in 'a b x y z'.split()]
        invariant_loop1 = And(z + x * y == a * b)
        doc_invariant = invariant_loop1

    if name == 'ps2':
        z3_vars = {v: Int(v) for v in 'x y c k'.split()}
        x, y, c, k = [z3_vars[v] for v in 'x y c k'.split()]
        invariant_loop1 = And(2 * x - y * y - y == 0, c == y, c <= k)
        doc_invariant = invariant_loop1

    if name == 'ps3':
        z3_vars = {v: Int(v) for v in 'x y c k'.split()}
        x, y, c, k = [z3_vars[v] for v in 'x y c k'.split()]
        invariant_loop1 = And(6 * x - 2 * y * y * y - 3 * y * y - y == 0, c == y, c <= k)
        doc_invariant = invariant_loop1

    if name == 'ps4':
        z3_vars = {v: Int(v) for v in 'x y c k'.split()}
        x, y, c, k = [z3_vars[v] for v in 'x y c k'.split()]
        invariant_loop1 = And(4 * x - y * y * y * y - 2 * y * y * y - y * y == 0, c == y, c <= k)
        doc_invariant = invariant_loop1

    if name == 'ps5':
        z3_vars = {v: Int(v) for v in 'x y c k x0 y0'.split()}
        x, y, c, k, x0, y0 = [z3_vars[v] for v in 'x y c k x0 y0'.split()]
        invariant_loop1 = And(6 * y * y * y * y * y + 15 * y * y * y * y+ 10 * y * y * y - 30 * x - y == 0, c == y, c <= k)
        doc_invariant = invariant_loop1
        invariant += [x0 == 0, y0 == 0]

    if name == 'ps6':
        z3_vars = {v: Int(v) for v in 'x y c k x0 y0'.split()}
        x, y, c, k, x0, y0 = [z3_vars[v] for v in 'x y c k x0 y0'.split()]
        invariant_loop1 = And(-2 * y * y * y * y * y * y - 6 * y * y * y * y * y -5 * y * y * y * y + y * y + 12 * x == 0, c == y, c <= k)
        doc_invariant = invariant_loop1
        invariant += [x0 == 0, y0 == 0]

    if name == 'sqrt1':
        z3_vars = {v: z3.Int(v) for v in 'a n t s'.split()}
        a, n, t, s = (z3_vars[v] for v in 'a n t s'.split())
        invariant_loop1 = And(a * a <= n, t == 2 * a + 1, s == (a + 1) * (a + 1))
        doc_invariant = invariant_loop1

    if name == 'knuth':
        z3_vars = {v: z3.Int(v) for v in 'd q r k n'.split()}
        d, q, r, k, n = (z3_vars[v] for v in 'd q r k n'.split())
        doc_invariant = And(d*d*q - 4*r*d + 4*k*d - 2*q*d + 8*r == 8*n, d % 2 == 1)

    s = Solver()
    s.set("timeout", 10_000)
    invariant = z3.simplify(z3.And(invariant))
    # if ctx_simplify and (name in ('divbin', 'cohendiv', 'cohencu', 'mannadiv', 'hard', 'prodbin', 'prod4br', 'freire1') or (name in ('dijkstra') and loop_idx == 1)):
    #     invariant = z3.Tactic("ctx-solver-simplify")(invariant).as_expr()
    #     doc_invariant = z3.Tactic("ctx-solver-simplify")(doc_invariant).as_expr()
    s.add(Not(Implies(invariant, doc_invariant)))
    res = s.check()
    if res == z3.unknown:
        print('Check timout! (Invariant probably incorrectut)')
    return res==z3.unsat, invariant, doc_invariant
