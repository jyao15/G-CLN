import numpy as np
import pandas as pd
from fractions import Fraction
from math import gcd, floor
import torch
import deepinv as dinv
from sklearn.preprocessing import normalize
from subprocess import run
import json
# from new_multi_lin_eq_solver import CLN


# max_denominator = 10
max_denominator = 2


class CLN(torch.nn.Module):
    def __init__(self, inputSize, midSize):
        super(CLN, self).__init__()
        self.inputSize, self.midSize = inputSize, midSize
        self.or_gates = torch.nn.Parameter(torch.Tensor(midSize, inputSize // midSize).uniform_(1.0))
        self.and_gates = torch.nn.Parameter(torch.Tensor(midSize).fill_(1.0))

    def forward(self, x):
        # print(x.shape)
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


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(outputSize, inputSize).uniform_(-1, 1))
        # self.weight = torch.nn.Parameter(torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            for weight in self.weight:
                weight /= torch.max(torch.abs(weight))
        out = torch.nn.functional.linear(x, self.weight)
        return out


def data_normalize(data):
    data = 10 * normalize(data, norm='l2', axis=1)
    return data


def gcln_infer(problem_number, learning_rate=0.01, max_epoch=4000,
                 loss_threshold=1e-6, non_loop_invariant=None):
    df = pd.read_csv("../benchmarks/code2inv/traces/" + str(problem_number) + ".csv")
    df_data = df.drop(columns=['init', 'final'])
    df_data['1'] = 1
    consts = dinv.load_consts(problem_number, '../benchmarks/code2inv/smt2/const.txt')

    Is = gcln_infer_data(df_data, consts, learning_rate=learning_rate, max_epoch=max_epoch,
            loss_threshold=loss_threshold, non_loop_invariant=non_loop_invariant, pname=problem_number)

    ext = ".c"

    run(['mkdir', '-p', '../benchmarks/code2inv/tmp'])

    for I in Is:
        p = dinv.smt_check(I.sexpr(), str(problem_number) + ext, "../benchmarks/code2inv/tmp", 
                "../benchmarks/code2inv/smt2")
        if p is None:
            continue
        screen_output = p.stdout.decode("utf-8")
        solved = screen_output.count('unsat') == 3
        if solved:
            break

    return solved, I

   
    
def gcln_infer_data(df_data, consts, learning_rate=0.01, max_epoch=4000,
                 loss_threshold=1e-6, non_loop_invariant=None, min_std=0.1, max_denominator = 10, pname=1):
    data = df_data.to_numpy(dtype=np.float) 
    data = np.unique(data, axis=0)
    data = data_normalize(data)

    # or_reg=(0.0000001, 1.00001, 0.0000001)
    or_reg=(0.001, 1.00001, 0.1)
    # and_reg=(1.000, 0.99999, 0.1)
    and_reg=(1.0, 0.99999, 0.1)

    or_reg, or_reg_decay, max_or_reg = or_reg
    and_reg, and_reg_decay, min_and_reg = and_reg

    ges, les, eqs = dinv.infer_single_var_bounds_consts(df_data, consts)

    input_size = data.shape[1]
    coeff = None
    if input_size > 1:
        valid_equality_found = False

        # data preparation
        mid_width, out_width = 1, 1
        inputs_np = np.array(data, copy=True)
        means_input, std_input = np.zeros([input_size], dtype=np.double), np.zeros([input_size], dtype=np.double)
        for i in range(input_size):
            means_input[i] = np.mean(data[:, i])
            std_input[i] = np.std(data[:, i])
            inputs_np[:, i] = (data[:, i] - means_input[i])
        inputs = torch.from_numpy(inputs_np).float()

        # build and train the model
        input_size = input_size

        model = linearRegression(input_size, 1)
        cln = CLN(mid_width, out_width)
        optimizer = torch.optim.Adam(list(model.parameters())+list(cln.parameters()), lr=learning_rate)
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            outputs_std = max([outputs.std().detach(), min_std])
            activation = dinv.gaussian(outputs, outputs_std)
            final_outputs = cln(activation.reshape([-1,1]))
            main_loss = 1 - final_outputs.mean()

            or_reg = min(or_reg * or_reg_decay, max_or_reg)
            and_reg = max(and_reg * and_reg_decay, min_and_reg)
            l_or_reg =  or_reg * torch.sum(torch.abs(cln.or_gates))
            l_and_reg =  -and_reg * torch.sum(torch.abs(cln.and_gates))

            loss = main_loss + l_or_reg + l_and_reg 

            if main_loss < loss_threshold:
                valid_equality_found = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cln.parameters(), 0.01)
            optimizer.step()

        # calculate final coeff
        # if valid_equality_found:
        coeff_ = model.weight.detach().numpy().reshape([input_size])
        scaled_coeff = np.round(coeff_/np.abs(coeff_).min())
        coeff = []
        denominator = 1
        for i in range(input_size):
            a = Fraction.from_float(float(coeff_[i])).limit_denominator(max_denominator)
            coeff.append(a)
            denominator = denominator * a.denominator // gcd(denominator, a.denominator)
        coeff = np.asarray([[floor(a * denominator) for a in coeff]])


    var_names = list(df_data.columns)
    
    # print('loading', pname)
    with open('../benchmarks/code2inv/conditions/' + str(pname) + '.json', 'r') as f:
        condition = json.load(f)
    # print(condition)

    pred = condition['predicate']

    Is = dinv.construct_invariant(var_names, coeff, ges, les, eqs, pred, non_loop_invariant)
    if scaled_coeff.max() < 50: # large coeffs cause z3 timeouts
        scaled_Is = dinv.construct_invariant(var_names, scaled_coeff.reshape(1,-1), ges, les, eqs, pred, non_loop_invariant)
        Is.extend(scaled_Is)
    return Is
    # ext = ".c"
    # if mod:
        # ext = "_mod"
    # p = dinv.smt_check(I.sexpr(), str(problem_number) + ext, "../results", "../code2inv/smt2")
    # screen_output = p.stdout.decode("utf-8")
    # if v:
        # print(screen_output)
        # print()
    # return screen_output.count('unsat') == 3


if __name__ == '__main__':
    import sys
    problem_number = int(sys.argv[1])
    # problem_number = 1
    gcln_infer(problem_number)
