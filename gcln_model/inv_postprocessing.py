import numpy as np
from z3 import *
from functools import reduce


def filter_coeffs(coeffs, and_gates, or_gates, inputs):
    # filter valid invariants by and_gates, or_gates and fitting error
    data_size, or_span, and_span = inputs.shape[0], or_gates.shape[1], len(and_gates)
    and_to_remain = []
    or_to_remain = [[] for _ in range(and_span)]
    for j, and_gate, or_gate_, coeff_ in zip(range(and_span), and_gates, or_gates, coeffs):
        if and_gate < 0.9 or np.max(or_gate_) < 0.9:
            continue
        and_to_remain.append(j)
        errors = []
        for i, or_gate, coeff in zip(range(or_span), or_gate_, coeff_):
            if or_gate < 0.9:
                continue
            or_to_remain[j].append(i)
            errors.append(np.abs(np.matmul(inputs, coeff.reshape(-1, 1)).squeeze()))
        errors = np.asarray(errors, dtype=np.float)
        error = np.mean(np.min(errors, axis=0))
        # print('And', j, '   Error:', error)
        if error > 1e-6:
            and_to_remain.pop()

    result = []
    for i, ors in zip(and_to_remain, or_to_remain):
        for j in or_to_remain[i]:
            # print(coeffs[i,j,:])
            result.append(coeffs[i,j,:].copy())

    return result


def decompose_coeffs(filtered_coeffs):
    # de-mixture and deduplicate the invariants
    filtered_coeffs = np.asarray(filtered_coeffs, dtype=np.int)
    if len(filtered_coeffs) == 0:
        # no invariant learned, return empty
        return filtered_coeffs

    if len(filtered_coeffs.shape) < 2:
        filtered_coeffs = filtered_coeffs.reshape(1, -1)

    num_inv, num_terms = filtered_coeffs.shape
    to_remove_indices = []
    for i in range(num_inv):
        for j in range(i+1, num_inv):
            if np.array_equal(filtered_coeffs[i, :], filtered_coeffs[j, :]) or np.array_equal(filtered_coeffs[i, :], - filtered_coeffs[j, :]):
                to_remove_indices.append(j)
    filtered_coeffs = np.delete(filtered_coeffs, to_remove_indices, 0)
    basic_coeffs = gaussian_elimination(filtered_coeffs)
    return basic_coeffs


def gaussian_elimination(A):
    ERR = 1e-6
    n, m = A.shape
    A = A.astype(np.float)

    first_non_zero_column = 0
    rank = np.minimum(n,m)
    for i in range(0, np.minimum(n,m)):
        maxEl, maxRow = 0, None
        while first_non_zero_column < m:
            # Search for maximum in this column
            maxEl = abs(A[i][first_non_zero_column])
            maxRow = i
            for k in range(i + 1, n):
                if abs(A[k][first_non_zero_column]) > maxEl:
                    maxEl = abs(A[k][first_non_zero_column])
                    maxRow = k
            if maxEl > ERR:
                break
            first_non_zero_column += 1

        if first_non_zero_column == m and maxEl <= ERR:
            # current row and all rows below are all zero
            rank = i
            break

        # Swap maximum row with current row (column by column)
        for k in range(first_non_zero_column, m):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # normalize current row s.t. first non_zero element is one
        A[i, :] = A[i, :] / A[i][first_non_zero_column]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -A[k][first_non_zero_column] / A[i][first_non_zero_column]
            for j in range(first_non_zero_column, m):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

        first_non_zero_column += 1

    # now A is upper-triangular, we will make it diagonal-like
    for i in range(rank - 1, -1, -1):
        first_non_zero_column = 0
        while A[i][first_non_zero_column] == 0:
            first_non_zero_column += 1
        for k in range(0, i):
            c = -A[k][first_non_zero_column] / A[i][first_non_zero_column]
            for j in range(first_non_zero_column, m):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # round elements to integers
    for i in range(rank):
        fraction_list = [Fraction.from_float(float(A[i][j])).limit_denominator(100000) for j in range(m)]
        numerator_list, denominator_list = [a.numerator for a in fraction_list], [a.denominator for a in fraction_list]

        def gcd_zero_considered(a, b):
            if a == 0:
                return b
            elif b == 0:
                return a
            return np.gcd(a, b)
        numerator_gcd = reduce(gcd_zero_considered, numerator_list)
        denominator_lcm = reduce(np.lcm, denominator_list)
        A[i, :] = denominator_lcm * A[i, :] / numerator_gcd
    A = A[:rank, :].round().astype(np.int)
    return A
            

def construct_eq(eq_coeff, var_names):
    reals = []
    for var in var_names:
        reals.append(Real(var))

    print(reals)
        
    eq_constraint = 0 * 0
    if (eq_coeff[0] != 0):
        eq_constraint = reals[0]*eq_coeff[0]
    for i, real in enumerate(reals[1:]):
        if ( eq_coeff[i+1] != 0):
            eq_constraint += eq_coeff[i+1] * real

    return eq_constraint == 0

     
def construct_eq_str(eq_coeff, var_names):
    constr = []
    for c, v in zip(eq_coeff, var_names):
        if c != 0:
            constr.append(str(c)+v)
    return ' + '.join(constr) + ' == 0'


def get_syms(expr):
    if not expr.children():
        if not(str(expr).isnumeric()):
            return [(str(expr), expr)]
        else:
            return []
    else:
        syms = [get_syms(c) for c in expr.children()]
        return [l for sym in syms for l in sym]
    
def parse_var_name(name, z3_vars):
    if isinstance(name, str):
        if 'GCD' in name:
            z3_vars[name] = z3.Int(name)
            return z3_vars

        if name == '(* 1)':
            z3_vars[name] = IntVal(1)
            return z3_vars
        if name.isdigit():
            z3_vars[name] = IntVal(int(name))
            return z3_vars

        n = name.strip('()*%')
        nn = n.split()
        for var in nn:
            if var not in z3_vars:
                if var.isdigit():
                    z3_vars[var] = IntVal(int(var))
                else:
                    z3_vars[var] = z3.Int(var)

        if '%' in name:
            fullvar = z3_vars[nn[0]] % z3_vars[nn[1]]
        else:
            fullvar = z3_vars[nn[0]]
            for var in nn[1:]:
                fullvar *= z3_vars[var]
        z3_vars[name] = fullvar
    else:
        # if just float
        z3_vars[name] = float(name)

    return z3_vars

def parse_simple(simple_inv, z3_vars):
    if len(simple_inv) == 3:
        rhs, lhs, pred = simple_inv
        additional = 0
    else:
        rhs, lhs, pred, additional = simple_inv
        additional = int(additional)
    parse_var_name(rhs, z3_vars)
    parse_var_name(lhs, z3_vars)
    
    if '==' in pred:
        return z3_vars[rhs] == z3_vars[lhs] + additional
    elif '<=' in pred:
        return z3_vars[rhs] <= z3_vars[lhs] + additional
    elif '<' in pred:
        return z3_vars[rhs] < z3_vars[lhs] + additional
    elif '>=' in pred:
        return z3_vars[rhs] >= z3_vars[lhs] + additional
    elif '>' in pred:
        return z3_vars[rhs] > z3_vars[lhs] + additional
    elif '!=' in pred:
        return z3_vars[rhs] != z3_vars[lhs] + additional
    else:
        raise ValueError("invalid predicate for 2 var invariant")
        

def compose_invariant(simple_invs, coeffs, names, ineq_invs, problem):
    inv = []
    
    z3_vars = {}

    for ii in ineq_invs:
        syms = get_syms(ii)
        for s in syms:
            if not s[0] in z3_vars:
                z3_vars[s[0]] = s[1]
        inv.append(ii)
        
    for name in names:
        if not name in z3_vars:
            parse_var_name(name, z3_vars)

    for coeff in coeffs:
        if len(coeff) > 0:
            eq_inv = coeff[0] * z3_vars[names[0]]
            for i in range(1, len(coeff)):
                try:
                    eq_inv += coeff[i]*z3_vars[names[i]]
                except:
                    print(eq_inv)
                    print(coeff[i])
                    print(names[i])
                    print(coeff[i]*z3_vars[names[i]])
                    print(eq_inv + coeff[i]*z3_vars[names[i]])
                    exit(-1)
            inv.append(eq_inv == 0)
        
    for simple_inv in simple_invs:
        inv.append(parse_simple(simple_inv, z3_vars))
        
    return inv, z3_vars


if __name__ == '__main__':
    # test_arr = np.array([[1,2,3,4], [3,6,-6,-8], [-1,-2,-3,-4], [2,4,21,28]])
    test_arr = np.array([[1,2], [2,4], [3,6]])
    print(decompose_coeffs(test_arr))
