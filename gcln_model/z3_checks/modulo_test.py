from z3 import *


# A, B, r, d, p, q = z3.Ints('A B r d p q')
# A2, B2, r2, d2, p2, q2 = z3.Ints('A2 B2 r2 d2 p2 q2')
# s = z3.Solver()
# s.add([Not(Implies(And(A >= 0,
#                  B >= 1,
#                  r >= 0,
#                  d >= 1,
#                  p >= 1,
#                  q >= 0,
#                  q <= 9,
#                  1*d + -1*r > 0,
#                  1*A + -1*q > -1,
#                  1*A + -1*r > -1,
#                  -1*B + 1*d > -1,
#                  p != 1,
#                  And(A2 == A,
#                      B2 == B,
#                      d2 == d/2,
#                      p == p/2,
#                      r >= d2,
#                      r2 == r - d2,
#                      q2 == q + p2)),
#              q2 <= 9))])
# print(s.check())



# x, y, q, r = Ints('x y q r')
# q1, r1 = Ints('q1 r1')
# test = Implies(And(x==q*y+r, x>0, y>0, r>=0, r<y), Exists([r1], And(x==q*y+r1, x>0, y>0, r1>=0, r1<y)))
# s = Solver()
# s.add(Not(And(test)))
# print(s.check())
# try:
#     print(s.model())
# except:
#     pass
