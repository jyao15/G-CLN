# coding: utf-8
import numpy as np
import pandas as pd
from fractions import Fraction
from math import gcd, floor
import torch
from tqdm import tqdm, tqdm_notebook


import deepinv as dinv
from poly_template_gen import setup_polynomial_data
from inv_postprocessing import filter_coeffs, decompose_coeffs


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, linear_bias):
        super(linearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(outputSize, inputSize).uniform_(-1, 1))
        # self.weight = torch.nn.Parameter(torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True))
        self.linear_bias = linear_bias
        if linear_bias:
            self.bias = torch.nn.Parameter(torch.zeros(outputSize))

    def forward(self, x, term_gates):
        with torch.no_grad():
            self.weight.data = self.weight * term_gates
            for weight in self.weight:
                weight /= torch.max(torch.abs(weight))
        if self.linear_bias:
            out = torch.nn.functional.linear(x, self.weight, self.bias)
        else:
            out = torch.nn.functional.linear(x, self.weight)
        return out


class CLN(torch.nn.Module):
    def __init__(self, inputSize, midSize):
        super(CLN, self).__init__()
        self.inputSize, self.midSize = inputSize, midSize
        self.or_gates = torch.nn.Parameter(torch.Tensor(midSize, inputSize // midSize).uniform_(0, 1))
        self.and_gates = torch.nn.Parameter(torch.Tensor(midSize).uniform_(0, 1))

    def forward(self, x):
        xs = torch.chunk(x, self.midSize, dim=1)
        with torch.no_grad():
            self.or_gates.data.clamp_(0.0, 1.0)
            self.and_gates.data.clamp_(0.0, 1.0)
        mids = []
        for x_, or_gate in zip(xs, self.or_gates):
            mid = 1 - torch.prod(1 - x_ * or_gate, -1)
            mids.append(mid.view(-1, 1))
        mids_ = torch.cat(mids, 1)
        # out = torch.prod(mids_, -1)
        out = torch.prod(1 + self.and_gates * (mids_ - 1), -1)
        return out


def mle_solve(problem_number, and_span=2, or_span=2, or_reg=(0.001, 1.001, 0.1), and_reg=(1.0, 0.999, 0.1), linear_bias=False, max_denominator=10, decay=1.0,
              growth_pruning=False, data_cleaning_threshold=-1, pruning_threshold=0.1, pre_sample=False, gen_poly=True, var_deg=None,
              max_deg=2, max_relative_degree=None, learning_rate=0.01, dropout=0.3, max_epoch=5000, loss_threshold=0.0, min_std=0.01, v=True, normalization=True,
             max_poly_const_terms=1, limit_poly_terms_to_unique_vars=False, drop_high_order_consts=False, loop_index=1, simple_invariant=False, lift_redundancy_deg=False, fractional_sampling=True, csv_path='../../numinv/nla/csv/', trace_path='../../traces/'):
    simple_invariants_ = []
    if type(problem_number) == int:
        df = pd.read_csv("../traces/" + str(problem_number) + ".csv", skipinitialspace=True)
        df_data = df.drop(columns=['trace_idx', 'init', 'final'], errors='ignore')
    else:
        if pre_sample:
            df = pd.read_csv(csv_path + str(problem_number) + ".csv", skipinitialspace=True)
            if fractional_sampling is True and problem_number in ['ps5', 'ps6']:
                _, __, simple_invariants_ = setup_polynomial_data(df.drop(columns=['trace_idx']), gen_poly=False, problem_number=problem_number)
                df = pd.read_csv(csv_path + str(problem_number) + "_fractional.csv", skipinitialspace=True)
        else:
            df = pd.read_csv(csv_path + str(problem_number) + '_' + str(loop_index) + ".csv", skipinitialspace=True)
        df_data = df[df['trace_idx'] == loop_index].drop(columns=['trace_idx'])
    # print("data: \n", df_data)
    data, var_names, simple_invariants = setup_polynomial_data(df_data, growth_pruning=growth_pruning,
            data_cleaning_threshold=data_cleaning_threshold, pruning_threshold=pruning_threshold,
            gen_poly=gen_poly, var_deg=var_deg, max_deg=max_deg, max_relative_degree=max_relative_degree,
            max_poly_const_terms=max_poly_const_terms, normalization=normalization,
            limit_poly_terms_to_unique_vars=limit_poly_terms_to_unique_vars, problem_number=problem_number,
            drop_high_order_consts=drop_high_order_consts, lift_redundancy_deg=lift_redundancy_deg, v=v)
    simple_invariants += simple_invariants_
    # print(var_names)
    # print('constant terms:', const_dict)
    # print('redundant terms:', redundancy_list)
    if simple_invariant:
        return simple_invariants, [], var_names

    final_result = multi_lin_eq_solve(data, and_span=and_span, or_span=or_span, or_reg=or_reg, and_reg=and_reg, linear_bias=linear_bias,
            max_denominator=max_denominator, learning_rate=learning_rate, dropout=dropout, decay=decay,
                       max_epoch=max_epoch, loss_threshold=loss_threshold, min_std=min_std, v=v)

    return simple_invariants, final_result, var_names


def multi_lin_eq_solve(data, and_span=2, or_span=2, or_reg=(0.001, 1.001, 0.1), and_reg=(1.0, 0.999, 0.1),
                       l2_reg=1e-10, lsqrt_reg=1e-10, max_denominator=10, linear_bias=False,
                       learning_rate=0.01, decay=1.0, max_epoch=50, loss_threshold=0.0, min_std=0.1, dropout=0.2, log_polyfit=False, v=False):
    data_size, num_terms = data.shape[0], data.shape[1]
    or_reg, or_reg_decay, max_or_reg = or_reg
    and_reg, and_reg_decay, min_and_reg = and_reg
    loss_hist = []
    debug_trace = []
    std_trace = []
    polyfit_hist = []
    if num_terms > 1:
        # data preparation
        inputs_np = np.array(data, copy=True)
        inputs_np_unshifted = inputs_np.copy()
        means_input, std_input = np.zeros([num_terms], dtype=np.double), np.zeros([num_terms], dtype=np.double)
        for i in range(num_terms):
            means_input[i] = np.mean(data[:, i])
            std_input[i] = np.std(data[:, i])
            inputs_np[:, i] = (data[:, i] - means_input[i])
        inputs = torch.from_numpy(inputs_np).float()

        # build and train the model
        term_gates = torch.tensor(np.random.binomial(n=1, p=1-dropout, size=(and_span * or_span, num_terms)), requires_grad=False, dtype=torch.float)
        model = linearRegression(num_terms, and_span * or_span, linear_bias=linear_bias)
        cln = CLN(and_span * or_span, and_span)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(cln.parameters()), lr=learning_rate)
        if v:
            print('initial model weights:', model.weight.detach().numpy().round(2))

        if v:
            epoch_iter = range(max_epoch)
        elif is_notebook():
            epoch_iter = tqdm_notebook(range(max_epoch))
        else:
            epoch_iter = tqdm(range(max_epoch))

        for epoch in epoch_iter:
            optimizer.zero_grad()
            linear_outputs = model(inputs, term_gates)
            std_vec = torch.std(linear_outputs.detach(), dim=0)
            outputs_std = torch.max(std_vec, torch.tensor([min_std]).expand_as(std_vec))
            activation = dinv.gaussian(linear_outputs, outputs_std)
            final_outputs = cln(activation)

            main_loss = 1 - final_outputs.mean()
            or_reg = min(or_reg * or_reg_decay, max_or_reg)
            and_reg = max(and_reg * and_reg_decay, min_and_reg)
            l_or_reg =  or_reg * torch.sum(torch.abs(cln.or_gates))
            l_and_reg =  -and_reg * torch.sum(torch.abs(cln.and_gates))

            l_2_norm_reg = l2_reg * torch.norm(model.weight, p=2) 
            if torch.isnan(l_2_norm_reg):
                l_2_norm_reg = 0.0

            l_sqrt_norm_reg = lsqrt_reg * torch.norm(model.weight.abs(), p=0.9)
            if torch.isnan(l_sqrt_norm_reg):
                l_sqrt_norm_reg = 0.0

            # print(main_loss, l_or_reg, l_and_reg, l_2_norm_reg, l_sqrt_norm_reg)
            loss = main_loss + l_or_reg + l_and_reg + l_2_norm_reg #+ l_sqrt_norm_reg
            # loss = main_loss + l_or_reg + l_and_reg + l_2_norm_reg + 0.0

                        # loss = main_loss + l_or_reg + l_and_reg 
            if torch.isnan(loss):
                # print('Instability detected, please rerun this problem.')
                return []
            loss.backward()
            optimizer.step()


            loss_hist.append(main_loss.item())
            debug_trace.append((main_loss.item(), l_2_norm_reg.item(), l_sqrt_norm_reg.item()))
            std_trace.append(outputs_std.data.numpy())
            if log_polyfit:
                polyfit = model.weight.matmul(inputs.t()).detach().abs().mean(1).flatten().numpy()
                polyfit_hist.append(polyfit)


            # apply decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay

            if v and epoch % 100 == 0:
                print('epoch {}, main_loss {}, loss {}'.format(epoch, main_loss.item(), loss.item()))
                print("standard deviation: ", outputs_std)
                # print('model weights grad:', model.weight.grad)
                # print('model weights:', torch.Tensor(model.weight.detach()))
                # print('cln or weights grad:', cln.or_gates.grad.data.numpy().flatten().round(2))
                print('cln or weights:', torch.Tensor(cln.or_gates.detach()).numpy().flatten().round(2))
                # print('cln and weights grad:', cln.and_gates.grad.data.numpy().round(2).flatten())
                print('cln and weights:', torch.Tensor(cln.and_gates.detach()).numpy().round(2).flatten())
                #print('final_output:', torch.Tensor(final_outputs.detach()))

            if epoch > 0 and abs(loss_hist[-1] - loss_hist[-2]) < loss_threshold:
                print('early stop at epoch {}, loss = {:0.4f}, diff_loss = {}'.format(epoch, loss_hist[-1],
                        abs(loss_hist[-1] - loss_hist[-2])))
                break

        # calculate final coeff
        if True:  # valid_equality_found:
            model.weight.data = model.weight * term_gates
            for weight in model.weight:
                weight /= torch.max(torch.abs(weight))
            coeff_ = model.weight.detach().numpy()
            or_gates = cln.or_gates.detach().numpy()
            and_gates = cln.and_gates.detach().numpy()

            coeffs = []
            for eq in range(and_span * or_span):
                coeff = []
                denominator = 1
                for i in range(num_terms):
                    a = Fraction.from_float(float(coeff_[eq][i])).limit_denominator(max_denominator)
                    coeff.append(a)
                    denominator = denominator * a.denominator // gcd(denominator, a.denominator)
                coeff = np.asarray([[floor(a * denominator) for a in coeff]])
                coeffs.append(coeff)

    # print('Or gates:', or_gates, '\nAnd gates:', and_gates)
    # print(coeff_)
    #print('Normalized coeff:\n', coeffs)
    # final_guess = filter_coeffs(np.asarray(coeffs, dtype=np.float).reshape(and_span, or_span, num_terms), and_gates, or_gates, inputs_np_unshifted)
    filtered_coeffs = filter_coeffs(np.asarray(coeffs, dtype=np.float).reshape(and_span, or_span, num_terms), and_gates, or_gates, inputs_np_unshifted)
    final_guess = decompose_coeffs(filtered_coeffs)

    if log_polyfit:
        polyfit_hist = np.stack(polyfit_hist)
    if v: 
        print (coeffs)
 
    # return or_gates, and_gates, coeff_, coeffs, loss_hist, pd.DataFrame(debug_trace, columns=['loss', 'l_2', 'l_sqrt']), polyfit_hist
    # return  (or_gates, and_gates, coeff_, coeffs, loss_hist, (debug_trace, std_trace, polyfit_hist)), final_guess
    return  final_guess

