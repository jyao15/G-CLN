import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
import torch
import deepinv as dinv
import itertools
from fractions import Fraction
from z3 import *
import z3
from functools import reduce
from tqdm import tqdm
from z3_checks import get_z3_checks, gen_rec_constraints
from poly_template_gen import get_consts
from math import gcd, floor
from inv_postprocessing import compose_invariant


def greater_or_equal(x, C1, C2):
    y_pos = 1 / (1 + (x / C2) ** 2)
    y_neg = 1 / (1 + (x / C1) ** 2)
    pos_mask = x.ge(0.0).float()
    y = y_pos * pos_mask + y_neg * (1 - pos_mask)
    return y


class gt_2d_diamond_model(torch.nn.Module):
    def __init__(self, C1=1.0, C2=25.0, weight_reg=True):
        super(gt_2d_diamond_model, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor([[1.0, -1.0],
                                                  [-1.0, 1.0],
                                                  [1.0, 1.0],
                                                  [-1.0, -1.0]]))
        self.b  = torch.nn.Parameter(torch.tensor([0.0,0.0,0.0,0.0]))
        self.C1, self.C2 = C1, C2
        self.weight_reg = weight_reg
        
    def forward(self, x):
        if self.weight_reg:
            with torch.no_grad():
                for weight in self.W:
                    weight /= torch.max(torch.abs(weight))
        linear = torch.matmul(x, self.W.t()) + self.b
        out = greater_or_equal(linear, self.C1, self.C2)
        return out
    
    def plot(self, X, title='', thresh=-10000):
        if thresh:
            out = self.forward(X)
            bound_scores = out.detach().log().sum(0).numpy()
            valid_bounds = bound_scores > thresh


        x = torch.tensor(sorted(list(set(X[:,0]))))
        plt.figure()
        plt.plot(X[:,0], X[:,1], '.')
        plt.title(title)
        for i in range(self.W.shape[0]):
            y = -self.W[i,0]/self.W[i,1]*x - self.b[i]/self.W[i,1]
            xp = x.detach().numpy()
            yp = y.detach().numpy()
            if valid_bounds[i]:
                plt.plot(xp, yp, 'r')
            else:
                plt.plot(xp, yp, 'r--')


class gt_3d_diamond_model(torch.nn.Module):
    def __init__(self, C1=1, C2=50.0, weight_reg=True):
        super(gt_3d_diamond_model, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor([[1.0, 1.0, -1.0],
                                                  [1.0, -1.0, 1.0],
                                                  [-1.0, 1.0, 1.0],
                                                  [-1.0, -1.0, 1.0],
                                                  [-1.0, 1.0, -1.0],
                                                  [1.0, -1.0, -1.0]]))
        self.b = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.C1, self.C2 = C1, C2
        self.weight_reg = weight_reg

    def forward(self, x):
        if self.weight_reg:
            with torch.no_grad():
                for weight in self.W:
                    weight /= torch.max(torch.abs(weight))
        linear = torch.matmul(x, self.W.t()) + self.b
        out = greater_or_equal(linear, self.C1, self.C2)
        return out


def basic_infer_2d_bounds(data, var_pair=[], v=0, max_iters=500, weight_reg=True):
    # drop duplicates:
    data = np.unique(data, axis=0)

    # drop large values:
    mask = np.min(np.abs(data) < 100, axis=1)
    data = data[mask, :]
    
    X = torch.tensor(data, dtype=torch.float)
    decay = 0.9995
    early_stop_thresh = 1e-6

    m = gt_2d_diamond_model(weight_reg=weight_reg)
    optimizer = torch.optim.Adam(params=[m.W, m.b], lr=0.1)

    loss_hist = []
    for i in range(max_iters):
        optimizer.zero_grad()
        out = m.forward(X)

        l_p = (1.0 - out).mean()
        loss = l_p

        loss_hist.append(loss.item())

        if (i+1) % 25 == 0:

            if abs(loss_hist[-1] - loss_hist[-2]) < early_stop_thresh:
                if v:
                    print('early stop at', i, loss_hist[-1])
                break

        if not loss.requires_grad or torch.isinf(loss):
            print('invalid loss', loss)
            break
        else:
            # loss.retain_grad()
            loss.backward()
            optimizer.step()
            for pg in optimizer.param_groups:
                pg['lr'] *= decay

    out = m.forward(X)

    if v:
        print('iters', i)
        print('W', m.W.data)
        print('b', m.b.data)
        print(out.log().sum(0).data)

    bound_scores = out.detach().log().sum(0).numpy()
    valid_bounds = [True for _ in bound_scores]
    
    # recover unshifted bounds
    coeffs = m.W.detach().numpy()
    bounds = m.b.detach().numpy()
    selected_coeffs = []
    selected_bounds = []
    for i, bound_valid in enumerate(valid_bounds):
        if bound_valid:
            # b = np.multiply(coeffs[i, :],means).sum() - bounds[i]
            b = bounds[i]
            selected_coeffs.append(coeffs[i, :])
            selected_bounds.append(b)
            
    return selected_coeffs, selected_bounds


def infer_3d_bounds(data, v=0, max_iters=500, weight_reg=True):
    # drop duplicates:
    data = np.unique(data, axis=0)

    X = torch.tensor(data, dtype=torch.float)
    decay = 0.9995
    early_stop_thresh = 1e-6

    m = gt_3d_diamond_model(weight_reg=weight_reg)
    optimizer = torch.optim.Adam(params=[m.W, m.b], lr=0.1)

    loss_hist = []
    for i in range(max_iters):
        optimizer.zero_grad()
        out = m.forward(X)

        l_p = (1.0 - out).mean()
        loss = l_p

        loss_hist.append(loss.item())

        if (i + 1) % 25 == 0:

            if abs(loss_hist[-1] - loss_hist[-2]) < early_stop_thresh:
                if v:
                    print('early stop at', i, loss_hist[-1])
                break

        if not loss.requires_grad or torch.isinf(loss):
            print('invalid loss', loss)
            break
        else:
            # loss.retain_grad()
            loss.backward()
            optimizer.step()
            for pg in optimizer.param_groups:
                pg['lr'] *= decay

    return m.W.data.detach().numpy(), m.b.data.detach().numpy()


def extract_3d_constraints(W, bs, z3_vars, var_names):
    W = W.copy()
    bs = bs.copy()
    bounds = []
    for i in range(len(W)):
        coeffs = W[i,:]
        b = bs[i]
        bound = build_gt_constr(coeffs, b, var_names, z3_vars)
        if bound is not None:
            bounds.append(bound)
    return bounds


def infer_1d_bounds(df, z3_vars, z3_quad_vars=set()):
    mins = df.min()
    maxs = df.max()
    bounds_1d = []
    for var in df.columns:
        if var in z3_vars and not var in z3_quad_vars:
            if z3_vars[var].is_int():
                bounds_1d.append(z3_vars[var] >= int(mins[var]))
                bounds_1d.append(z3_vars[var] <= int(maxs[var]))
            else:
                bounds_1d.append(z3_vars[var] >= mins[var])
                bounds_1d.append(z3_vars[var] <= maxs[var])
        
    return bounds_1d


def build_gt_constr(coeff_old, bound, var_names, z3_vars, max_denom=2):
    b_rounded = Fraction.from_float(float(bound)).limit_denominator(max_denom)
    # z3_vars = [Real(v) for v in var_names]
    coeff = [b_rounded]
    denominator = b_rounded.denominator
    for i in range(len(coeff_old)):
        a = Fraction.from_float(float(coeff_old[i])).limit_denominator(max_denom)
        coeff.append(a)
        denominator = denominator * a.denominator // gcd(denominator, a.denominator)
    coeff = np.asarray([floor(a * denominator) for a in coeff])
    b_rounded, coeff = coeff[0], coeff[1:]

    # single var inequality, should be handled separately
    if np.sum(coeff != 0) <= 1 or np.abs(b_rounded) > 1:
        return None

    constraint = []
    for c, v in zip(coeff, var_names):
        # print(c, v)
        constraint.append(c*z3_vars[v])
    constraint = reduce(lambda a,b: a+b, constraint)
    if constraint.is_int():
        b_rounded = int(b_rounded)

    constraint = (constraint + b_rounded >= 0)
    
    return constraint


def check_valid(bounds, bound2s, ref, ref2, pre, lc, rec, s):
    s.add(z3.Not(z3.And(
                        z3.Implies(pre, 
                                   z3.And(*bounds, z3.And(*ref))),
                        z3.Implies(z3.And(*bounds, z3.And(*ref), lc, rec), 
                                   z3.And(*bound2s, z3.And(*ref2)))
                       )
                )
         )
    res = s.check()
    return res

def check_bounds(lc, pre, rec, ref,
                 bounds, z3_vars, bound2s, z3_var2s,
                 v=0, progbar=False):
    
    all_z3_vars = z3_vars.copy()
    all_z3_vars.update(z3_var2s)

    ref2, _ = gen_rec_constraints(ref, z3_vars)

    s = z3.Solver()
    s2 = z3.Solver()
    s.set("timeout", 5000)
    s2.set("timeout", 5000)
    res = z3.sat

    # first check eq constraint won't cause timeouts:
    s.push()
    res = check_valid([], [], ref, ref2, pre, lc, rec, s)
    s.pop()
    if res == z3.unknown:
        ref, ref2 = [], []

    valid_bounds = []
    valid_bound2s = []
    for bound, bound2 in tqdm(list(zip(bounds, bound2s))):
        s.push()
        res = check_valid(valid_bounds+[bound], valid_bound2s+[bound2], ref, ref2, pre, lc, rec, s)
        if res == z3.unsat:
            valid_bounds.append(bound)
            valid_bound2s.append(bound2)
        elif res == z3.unknown:
            ref = []
            ref2 = []
        else:
            # print('model',s.model())
            pass
        s.pop()

    loop = range(len(bounds)+len(ref))
    if progbar:
        loop = tqdm(loop)

    for i in loop:

        s.push()
        res1 = check_valid(bounds, bound2s, ref, ref2, pre, lc, rec, s)

        if v:
            print('check\n',s, res)
        if res1 == z3.unsat:
            if progbar:
                loop.update(1)
            break
        if res1 == z3.unknown:
            s.pop()
            ref = []
            ref2 = []
            continue

        
        m = s.model()
        try:
            model_constraint = [all_z3_vars[str(mi)] == m[mi] for mi in m]
        except KeyError:
            for mi in m:
                all_z3_vars[str(mi)] = z3.Int(str(mi))
            model_constraint = [all_z3_vars[str(mi)] == m[mi] for mi in m]

        if v:
            print('model\n',m)

        s2.push()
        s2.add(model_constraint)

        if v:
            print('bounds')
            print(bounds)

        bound_sats = []
        for bound, bound2 in zip(bounds, bound2s):
            s2.push()
            s2.add(bound, bound2)
            res = s2.check()
            bound_sats.append(res)
            if v:
                print('\t',bound, res)
            s2.pop()


        pruned_bounds = []
        pruned_bound2s = []
        for bound, bound2, bound_sat in zip(bounds, bound2s, bound_sats):
            if bound_sat == z3.sat or bound in valid_bounds:
                pruned_bounds.append(bound)
                pruned_bound2s.append(bound2)
        
        if len(pruned_bounds) == len(bounds):
            pruned_ref, pruned_ref2 = [], []
            for eq_constr, eq_constr2 in zip(ref, ref2):
                s2.push()
                s2.add(eq_constr, eq_constr2)
                res = s2.check()
                if res == z3.sat:
                    pruned_ref.append(eq_constr)
                    pruned_ref2.append(eq_constr2)
            ref = pruned_ref
            ref2 = pruned_ref2

        bounds = pruned_bounds
        bound2s = pruned_bound2s


        s.pop()
        s2.pop()

    if res1 != z3.unsat:
        # could not verify soundness, do not return any bounds
        bounds = valid_bounds
        
    return bounds 



def basic_infer_all_1d_bounds(df, v=0):
    df = df.copy()
    # remove consts and other invalid cols:
    drop_vars = ['trace_idx', 'while_counter', 'run_id', '1']
    consts = get_consts(df.to_numpy(), list(df.columns), 
                        df.run_id.to_numpy())

    var_names = [v for v in df.columns if v not in drop_vars]

    # get quadratic terms for non constants:
    quad_vars = {}
    for var in df.columns:
        if var not in drop_vars and var not in consts:
            var2 = '(* '+var+' '+var+')' 
            df[var2] = df[var] * df[var]
            quad_vars[var] = var2

    var_pairs = list(itertools.product(var_names, repeat=2))
    for var in var_names:
        for var2_var in quad_vars.keys():
            if var != var2_var:
                var_pairs.append((var, quad_vars[var2_var]))

    var_pairs = [tuple(sorted(vp)) for vp in var_pairs if vp[0] != vp[1]]
    var_pairs = set(var_pairs)
    
    if v:
        print(var_pairs)


    # gen z3 var dict:
    z3_vars = {}
    for var in var_names:
        z3_vars[var] = z3.Int(var)

    constrs = []

    # 1d vars
    constrs += infer_1d_bounds(df, z3_vars, [])
    return constrs, z3_vars


def basic_infer_all_2d_bounds(df, v=0, progbar=False, weight_reg=True, max_ineq_deg=1):

    df = df.copy()
    # remove consts and other invalid cols:
    drop_vars = ['trace_idx', 'while_counter', 'run_id', '1']
    consts = get_consts(df.to_numpy(), list(df.columns), 
                        df.run_id.to_numpy())

    var_names = [v for v in df.columns if v not in drop_vars]

    # get quadratic terms for non constants:
    quad_vars = {}
    if max_ineq_deg==2:
        for var in df.columns:
            if var not in drop_vars and var not in consts:
                var2 = '(* '+var+' '+var+')' 
                df[var2] = df[var] * df[var]
                quad_vars[var] = var2

    var_pairs = list(itertools.product(var_names, repeat=2))
    for var in var_names:
        for var2_var in quad_vars.keys():
            if var != var2_var:
                var_pairs.append((var, quad_vars[var2_var]))

    var_pairs = [tuple(sorted(vp)) for vp in var_pairs if vp[0] != vp[1]]
    var_pairs = set(var_pairs)
    
    if v:
        print(var_pairs)

    # gen z3 var dict:
    z3_vars = {}
    for var in var_names:
        z3_vars[var] = z3.Int(var) # Real(var)
    for var in quad_vars:
        z3_vars[quad_vars[var]] = z3_vars[var] * z3_vars[var]
    # print(z3_vars)
    
    loop = var_pairs
    if progbar:
        loop = tqdm(var_pairs)

    constrs = []

    # 1d vars
    constrs += infer_1d_bounds(df, z3_vars, set(quad_vars.values()))

    # 2d vars
    for var_pair in loop:
        if v:
            print(var_pair)
        coeffs, bounds =\
            basic_infer_2d_bounds(df[list(var_pair)].to_numpy(), 
                    var_pair=var_pair, v=v, weight_reg=weight_reg)
        tmp_list = []
        for c, b in zip(coeffs, bounds):
            new_constr = build_gt_constr(c, b, var_pair, z3_vars)
            if new_constr is not None:
                tmp_list.append(new_constr)
        constrs.extend(tmp_list)

    return constrs, z3_vars


def infer_numinv_bounds(problem_name, loop_index=1, progbar=True, csv_path='../benchmarks/nla/csv/',
        weight_reg=True, max_ineq_deg=1, ineq=0, partial=[]):

    learned_eqs = [And(True)]
    if len(partial) >= 3:
        # print(partial)
        learned_eqs, _ = compose_invariant(partial[0], partial[1], partial[2], [], problem_name)

    if type(ineq) == list:
        df = pd.read_csv('../benchmarks/nla/csv/' + problem_name + '_extended_samples.csv', skipinitialspace=True)
        df = df[df.trace_idx == 2]
        df_data = df[ineq]
        W, b = infer_3d_bounds(df_data.to_numpy(), v=0, weight_reg=weight_reg)

        bounds, z3_vars = basic_infer_all_1d_bounds(df)
        bounds += extract_3d_constraints(W, b, z3_vars, ineq)

        bound2s, z3_var2s = gen_rec_constraints(bounds, z3_vars)

        lc, pre, rec, post, _, _ = get_z3_checks(problem_name, loop_index, z3_vars, z3_var2s)

        if progbar:
            print('Performing soundness checks (bar represents max possible checks, will finish early)')
        checked_bounds = check_bounds(lc, pre, rec, [z3.And(True)], bounds, z3_vars, bound2s, z3_var2s, progbar=True)
        return checked_bounds

    elif ineq == 1:

        df = pd.read_csv(csv_path + problem_name +'_'+str(loop_index)+ '.csv', skipinitialspace=True)
        df = df[df.trace_idx==loop_index]

        # drop large values:
        # df_data = df.drop(['trace_idx', 'while_counter', 'run_id', '1'], axis=1)
        # df = df[(df_data < 15).all(1)]
        bounds, z3_vars = basic_infer_all_2d_bounds(df, progbar=progbar, weight_reg=weight_reg, max_ineq_deg=max_ineq_deg)

        bound2s, z3_var2s = gen_rec_constraints(bounds, z3_vars)

        lc, pre, rec, post, _, _ = get_z3_checks(problem_name, loop_index, z3_vars, z3_var2s)

        if progbar:
            print('Performing soundness checks (bar represents max possible checks, will finish early)')
        checked_bounds = check_bounds(lc, pre, rec, learned_eqs, bounds, z3_vars, bound2s, z3_var2s, progbar=True)

        
    else:
        if ineq == -1:
            return []
        assert ineq == 0
        df = pd.read_csv(csv_path + problem_name +'_'+str(loop_index)+ '.csv', skipinitialspace=True)
        df = df[df.trace_idx==loop_index]

        # drop large values:
        df_data = df.drop(['trace_idx', 'while_counter', 'run_id', '1'], axis=1)
        # df = df[(df_data < 100).all(1)]
        df = df[(df_data < 15).all(1)]
        bounds, z3_vars = basic_infer_all_1d_bounds(df)

        bound2s, z3_var2s = gen_rec_constraints(bounds, z3_vars)

        lc, pre, rec, post, _, _ = get_z3_checks(problem_name, loop_index, z3_vars, z3_var2s)

        # print('precheck_bounds')
        # for b in bounds:
            # print('\t'+str(b))
        if progbar:
            print('Performing soundness checks (bar represents max possible checks, will finish early)')
        checked_bounds = check_bounds(lc, pre, rec, [And(True)], bounds, z3_vars, bound2s, z3_var2s, progbar=True)


    return checked_bounds
