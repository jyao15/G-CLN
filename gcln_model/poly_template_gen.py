import itertools
from collections import Counter
import numpy as np
from data_processing import *

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# "x2".translate(SUP) -> "x²"

def list_cross(l, num_cross=2):
    return set(map("".join, itertools.product(l, repeat=num_cross)))


def get_poly_deg(arr, while_counters, growth_pruning):
    if not growth_pruning:
        return 1
    max_index = np.argmax(while_counters)
    if while_counters[max_index] >= 5:
        degree = np.floor(np.log(abs(arr[max_index]) + 0.5) / np.log(while_counters[max_index]) + 0.5)
        return max(degree, 1)
    else:
        return 1


def get_poly_degs(data, var_names, growth_pruning, v=False):
    degs = {}
    while_counters = data[:, 0]
    for col in range(1, data.shape[1]):
        degs[var_names[col]] = get_poly_deg(data[:, col], while_counters, growth_pruning)
    if v:
        print("poly_degs: {}".format(degs))

    return degs

def get_consts(data, var_names, run_ids):
    consts = set()
    if run_ids is not None:
        for c in range(data.shape[1]):
            constant = True
            cur_val = data[0, c]
            cur_run_id = run_ids[0]
            for r in range(1, data.shape[0]):
                if run_ids[r] != cur_run_id:
                    cur_run_id = run_ids[r]
                    cur_val = data[r, c]
                elif cur_val != data[r, c]:
                    constant=False
                    break
            if constant:
                consts.add(var_names[c])
    else:
        # print('WARNING: attempting to identify const terms but there are no run_ids in the trace!')
        # print('         Rerun instrument.py on the problem c code to generate samples with run_ids.')
        pass
            
    return consts


'''
"var_degs" can be partial - if you want to manually define var "a" to be 3rd degree, 
simply pass in {"a": 3} and we'd still infer degs for all other variables.
'''


def get_poly_template(data, var_names, growth_pruning=False, run_ids=None, num_cross=None, max_deg=2, var_degs_assigned=None, v=False, max_relative_degree=None, max_poly_const_terms=1,
        limit_poly_terms_to_unique_vars=False, drop_high_order_consts=False):

    var_degs = get_poly_degs(data, var_names, growth_pruning, v=v)

    consts = get_consts(data, var_names, run_ids)
    if v:
        print('consts:', consts)

    for var in var_degs:
        if var in consts:
            var_degs[var] = 0
    if v:
        print('var degs:', var_degs)

    if var_degs_assigned is not None:
        var_degs.update(var_degs_assigned)
    if max_deg is None:
        max_deg = int(max(var_degs.values())) + max_relative_degree
    if v:
        print("max deg: {}".format(max_deg))

    vars = [""] + sorted(var_names[1:])
    if v:
        print("vars: {}".format(vars))

    # In theory, here we should use LCM(min(var_degs), max(var_degs))/min(var_degs)
    if num_cross is None:
        num_cross = max_deg
    poly_terms = list_cross(vars, num_cross)
    if v:
        print("unfiltered poly terms: {}".format(poly_terms))

    # add poly terms with all consts:
    consts_l = [''] + list(consts)
    consts_l_l = [consts_l for _ in range(max_poly_const_terms)]
    poly_consts = itertools.product(vars[1:], *consts_l_l)
    poly_consts = set(map(''.join, poly_consts))

    poly_terms = poly_terms.union(poly_consts)
    poly_terms = set(term for term in poly_terms if term.find('1') == -1)
    poly_terms.add('1')

    filtered_result = {}
    for term in poly_terms:
        term_str = []
        term_degs = Counter(term)
        term_deg = sum([term_degs[var] * var_degs[var] for var in term_degs.keys()])

        if v:
            print("{}, deg: {}".format(term_degs, term_deg))

        if term_deg > max_deg:
            continue

        if limit_poly_terms_to_unique_vars:
            # poly terms should only have at most 1 of each var
            if len(set(term)) > 1:
                if len(term) != len(set(term)):
                    continue
        
        if drop_high_order_consts:
            if len(set(term)) == 1:
                if term and term[0] in consts:
                    continue

        for var in sorted(term_degs.keys()):
            for _ in range(term_degs[var]):
                term_str.append(var)

        term_str = "(* {})".format(" ".join(term_str))

        filtered_result[term_str] = term_degs
    del(filtered_result['(* )'])
    return filtered_result


def poly_template_transform(data, var_names, run_ids=None, growth_pruning=False, num_cross=None, max_deg=2, var_degs=None, v=False, max_relative_degree=None, max_poly_const_terms=1, problem_number=0, drop_high_order_consts=False, limit_poly_terms_to_unique_vars=False):
    data_dict = {var_names[i]:data[:, i] for i in range(1, len(var_names))}

    terms_dict = get_poly_template(data, var_names, growth_pruning, run_ids=run_ids, num_cross=num_cross, max_deg=max_deg, var_degs_assigned=var_degs, v=v, max_relative_degree=max_relative_degree, max_poly_const_terms=max_poly_const_terms, limit_poly_terms_to_unique_vars=limit_poly_terms_to_unique_vars, drop_high_order_consts=drop_high_order_consts)
    if v:
        print("filtered poly terms: {}".format(", ".join(terms_dict.keys())))


    new_data, new_var_names = [], []
    term_tuple_list = sorted([(k, v) for k, v in terms_dict.items()])
    for i in range(len(term_tuple_list)):
        term_str, term_degs = term_tuple_list[i]
        new_var_names.append(term_str)
        new_col = np.ones(len(data), dtype=np.float)
        for var, var_deg in term_degs.items():
            new_col *= data_dict[var] ** var_deg
        new_data.append(new_col)
    if problem_number in ('egcd2', 'egcd3', 'lcm1', 'lcm2'):
        new_data.append(np.gcd(data_dict['x'].astype(int), data_dict['y'].astype(int)))
        new_data.append(np.gcd(data_dict['a'].astype(int), data_dict['b'].astype(int)))
        new_var_names += ['GCD(x,y)', 'GCD(a,b)']
    elif problem_number in ('fermat1', 'fermat2'):
        new_data.append(data_dict['u'].astype(int) % 2)
        new_data.append(data_dict['v'].astype(int) % 2)
        new_var_names += ['(% u 2)', '(% v 2)']
    new_data = np.asarray(new_data).transpose()
    return new_data, new_var_names


def get_simple_eq_invariants(data, var_names, problem_number):
    simple_eq_invariants = []
    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            if (np.array_equal(data[:, i], data[:, j])):
                simple_eq_invariants.append((var_names[i], var_names[j], ' == '))
    return simple_eq_invariants


def filter_const_invariants(invariants, const_dict):
    consts = list(const_dict.keys()) + ['1', '(* 1)']
    filtered_invariants = []
    for inv in invariants:
        if not ((inv[0] in consts) and (inv[1] in consts)):
            if len(inv[0]) <= 5 or len(inv[1]) <= 5 or (not inv[0].startswith('(*') and not inv[1].startswith('(*')):
                filtered_invariants.append(inv)
    return filtered_invariants


def setup_polynomial_data(df_data, growth_pruning=True, data_cleaning_threshold=100, pruning_threshold=0.1, gen_poly=True, var_deg=None, max_deg=None, max_relative_degree=None, max_poly_const_terms=1, limit_poly_terms_to_unique_vars=False, problem_number=0, drop_high_order_consts=False, lift_redundancy_deg=False, normalization=True, v=False):
    run_ids = None
    if 'run_id' in df_data.columns:
        run_ids = list(df_data.run_id)
        df_data = df_data.drop(columns=['run_id'])

    var_names = list(df_data.columns)
    data = df_data.to_numpy(dtype=np.float)
    if gen_poly:
        data, var_names = poly_template_transform(data, var_names, run_ids=run_ids, growth_pruning=growth_pruning, max_deg=max_deg, max_relative_degree=max_relative_degree, var_degs=var_deg, max_poly_const_terms=max_poly_const_terms, limit_poly_terms_to_unique_vars=limit_poly_terms_to_unique_vars, problem_number=problem_number, drop_high_order_consts=drop_high_order_consts, v=False)
    data = np.unique(data, axis=0)

    simple_eq_invariants = get_simple_eq_invariants(data, var_names, problem_number)
    # print('simple', simple_eq_invariants)
    data, const_dict = constant_check(data, var_names)
    filtered_simple_eq_invariants = filter_const_invariants(simple_eq_invariants, const_dict)
    filtered_simple_eq_invariants += [(key, value, ' == ') for key, value in const_dict.items() if (len(key) == 5 or key.startswith('(%'))]
    # print('simple fil', filtered_simple_eq_invariants)
    data, redundancy_list = redundancy_check(data, var_names)
    data, pruned_list = threshold_pruning(data, var_names, pruning_threshold)
 
    if v:
        print('Constant terms:', const_dict)
        print('Redundant terms:', redundancy_list)
        print('Unlikely terms:', pruned_list)
        print('Final candidate terms:', var_names)
    if data_cleaning_threshold > 1:
        data = remove_big_value(data, data_cleaning_threshold)
    if normalization:
        data = data_normalize(data)
    if v:
        print('Final training set size:', data.shape)
    if lift_redundancy_deg:
        _, _, new_filtered_simple_eq_invariants = setup_polynomial_data(df_data, problem_number=problem_number, max_deg=max_deg+1)
        filtered_simple_eq_invariants += new_filtered_simple_eq_invariants
    return data, var_names, filtered_simple_eq_invariants
