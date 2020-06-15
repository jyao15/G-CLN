import numpy as np
import datetime, time
import torch
import sys, getopt
import pandas as pd
from gcln_model import mle_solve, multi_lin_eq_solve
from gcln_code2inv import gcln_infer
from inv_postprocessing import compose_invariant
from sklearn.preprocessing import normalize
from checks import check_nla_invariant
import ineq_solver
import z3
from collections import defaultdict
import random

class config:
    def __init__(self, name, loop_index= 1, max_epoch=5000, 
            max_deg= 2, max_ineq_deg=1, and_span=10, or_span=1,
            pre_sample=False, decay=0.9996, max_relative_degree=None,
            and_reg=(1.0, 0.999, 0.1), or_reg=(0, 0, 0), var_deg=None, max_poly_const_terms=1,
            growth_pruning=False, dropout=0.3, max_denominator=10,
            drop_high_order_consts=False, normalization=True, weight_reg = True, min_std= 0.01, gen_poly=True,
            simple_invariant=False, lift_redundancy_deg=False, fractional_sampling=True,
            data_cleaning_threshold=-1,limit_poly_terms_to_unique_vars=False, ineq=0, seed=0):
        self.name = name
        self.max_epoch= max_epoch
        self.loop_index = loop_index
        self.max_deg = max_deg
        self.max_ineq_deg = max_ineq_deg
        self.and_span = and_span
        self.or_span = or_span
        self.pre_sample=pre_sample
        self.decay = decay
        self.growth_pruning=growth_pruning
        self.data_cleaning_threshold=data_cleaning_threshold
        self.dropout= dropout
        self.or_reg = or_reg
        self.and_reg = and_reg
        self.var_deg= var_deg
        self.max_denominator = max_denominator 
        self.max_relative_degree=max_relative_degree
        self.max_poly_const_terms=max_poly_const_terms
        self.limit_poly_terms_to_unique_vars=limit_poly_terms_to_unique_vars
        self.drop_high_order_consts= drop_high_order_consts
        self.lift_redundancy_deg = lift_redundancy_deg
        self.min_std= min_std
        self.normalization = normalization
        self.weight_reg = weight_reg
        self.fractional_sampling = fractional_sampling
        self.gen_poly = gen_poly
        self.simple_invariant = simple_invariant
        self.ineq = ineq
        self.seed = seed


    def run(self, random_seed=True):
        if random_seed:
            seed = random.randint(1, 10_000)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        else:
            seed = self.seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        print('Seed:', seed)

        # print('data normalization:', self.normalization, 'weight regularization:', self.weight_reg, 'dropout:', self.dropout, 'fractional sampling:', self.fractional_sampling)
        # self.max_epoch = 1
        self.res = mle_solve(self.name, loop_index = self.loop_index,
                    max_epoch = self.max_epoch, max_denominator = self.max_denominator,
                    max_deg = self.max_deg, and_span = self.and_span,
                    or_span = self.or_span, pre_sample= self.pre_sample,
                    decay = self.decay, growth_pruning = self.growth_pruning, 
                    or_reg = self.or_reg, and_reg = self.and_reg, var_deg=self.var_deg,
                    max_relative_degree = self.max_relative_degree,
                    max_poly_const_terms = self.max_poly_const_terms,
                    min_std = self.min_std, normalization=self.normalization, gen_poly = self.gen_poly,
                    limit_poly_terms_to_unique_vars = self.limit_poly_terms_to_unique_vars,
                    drop_high_order_consts = self.drop_high_order_consts,
                    lift_redundancy_deg = self.lift_redundancy_deg,
                    simple_invariant = self.simple_invariant, fractional_sampling = self.fractional_sampling,
                    dropout = self.dropout, data_cleaning_threshold=self.data_cleaning_threshold, v = False,
                    csv_path='../benchmarks/nla/csv/', trace_path='../benchmarks/nla/traces/')

        ineq_bounds = ineq_solver.infer_numinv_bounds(self.name, self.loop_index, progbar=True, csv_path='../benchmarks/nla/csv/', weight_reg=self.weight_reg, max_ineq_deg=self.max_ineq_deg, ineq=self.ineq, partial=self.res)

        if self.res is None:
            self.res = [[] for i in range(5)]
        
        # print(self.res)
        full_res = list(self.res) + [ineq_bounds]
        return full_res


#default = { 'loop_index': 1, 'max_deg':2, 'and_span':1, 'or_span':1,'pre_sample':True , 'decay':0.9996}
NAME = ['cohencu', 'cohendiv' , 'dijkstra', 'divbin', 'egcd', 'egcd2', 'egcd3', 'fermat1', 'fermat2', 'freire1', 'freire2' , 'geo1', 'geo2', 'geo3', 'hard', 'knuth', 'lcm1', 'lcm2', 'mannadiv', 'prod4br', 'prodbin', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'sqrt1']

PROBLEMS = {} 
for n in NAME:
    PROBLEMS[n] = config(n)
PROBLEMS = {'cohencu': [config ('cohencu', dropout=0.2, and_span=20, growth_pruning=True, lift_redundancy_deg=True, ineq=1)],
            'cohendiv': [config ('cohendiv', loop_index=1, pre_sample=True),
                config('cohendiv', loop_index=2, pre_sample=True, ineq=1)],
            'dijkstra': [config('dijkstra', loop_index=1, pre_sample=True, simple_invariant=True),
                       config ('dijkstra', loop_index=2, pre_sample=True, var_deg={'h': 3}, ineq=['r', 'p', 'q'])],
            'divbin': [config('divbin', loop_index=1),
                      config('divbin', loop_index=2, ineq=1)],
            'egcd': [config('egcd')],
            'egcd2': [config('egcd2', loop_index=1, growth_pruning=True, ineq=-1),
                      config('egcd2', loop_index=2, pre_sample=True, var_deg={'p':3, 'q':3, 'x':3, 'y':3, 'r':3, 's':3}, growth_pruning=True, ineq=-1)],
            'egcd3': [config('egcd3', loop_index=1, ineq=-1),
                    config('egcd3', loop_index=2, var_deg={'p':3, 'q':3, 'x':3, 'y':3, 'r':3, 's':3}, ineq=-1),
                    config('egcd3', loop_index=3, var_deg={'p':3, 'q':3, 'x':3, 'y':3, 'r':3, 's':3}, ineq=-1)],
            'fermat1': [config('fermat1', loop_index=1, dropout=0, data_cleaning_threshold=10000, var_deg={'A': 2, 'R': 3}),
                        config('fermat1', loop_index=2, dropout=0, data_cleaning_threshold=10000, var_deg={'A': 2, 'R': 3}),
                        config('fermat1', loop_index=3, dropout=0, data_cleaning_threshold=10000, var_deg={'A': 2, 'R': 3})],
            'fermat2': [config('fermat2', data_cleaning_threshold=10000, growth_pruning=True, var_deg={'A': 2, 'R': 3}, dropout=0)],
            'freire1': [config('freire1', and_span=20, growth_pruning=True)],
            'freire2': [config('freire2', max_deg=3, growth_pruning=True, and_span=20, max_denominator=15, pre_sample=True, dropout=0.2)],

            'geo1': [config('geo1', growth_pruning=True, and_span=20, var_deg={'c': 3, 'k': 3})],
            'geo2': [config('geo2', growth_pruning=True, and_span=20, var_deg={'c': 3, 'k': 3})],
            'geo3': [config('geo3', growth_pruning=True, max_epoch=10000, pre_sample=True, max_relative_degree=1, max_poly_const_terms=2, limit_poly_terms_to_unique_vars=True, and_span=20, var_deg={'c': 4, 'k': 4})],

            'hard': [config('hard', simple_invariant=True, loop_index=1),
                      config ('hard', loop_index=2, ineq=1)],
            'knuth': [config('knuth', max_deg=3, growth_pruning=True, max_poly_const_terms=0, ineq=-1)],

            'lcm1': [config('lcm1', loop_index=1, ineq=-1),
                     config('lcm1', loop_index=2, and_span=20, ineq=-1),
                     config('lcm1', loop_index=3, ineq=-1)],

            'lcm2': [config('lcm2')],
            'mannadiv': [config('mannadiv', dropout=0.1, growth_pruning=True, ineq=1)],
            'prod4br': [config('prod4br', max_deg=3, pre_sample=True, var_deg={'q':3, 'p':3, 'a':0, 'b':0, 'x':1, 'y':1}, drop_high_order_consts=True)],
            'prodbin' : [config('prodbin')],
            'ps2': [config('ps2', dropout=0, ineq=1, max_deg=2, growth_pruning=True, var_deg={'k': 10})],
            'ps3': [config('ps3', dropout=0, ineq=1, max_deg=3, growth_pruning=True, var_deg={'k': 10})],
            'ps4': [config('ps4', dropout=0, ineq=1, max_deg=4, growth_pruning=True, var_deg={'k': 10})],
            'ps5': [config('ps5', dropout=0, ineq=1, pre_sample=True, gen_poly=False, max_denominator=30, min_std=0.1)],
            'ps6': [config('ps6', dropout=0, ineq=1, pre_sample=True, gen_poly=False, max_denominator=15, min_std=0.1, and_span=20)],
            'sqrt1': [config('sqrt1', dropout=0.5, and_span=20, max_ineq_deg=2, ineq=1)]
            }


def get_inv_str(inv, var_names):
    inv_str = []
    if var_names:
        for t,v in zip(inv, var_names):
            if abs(t) > 1e-6:
                inv_str.append(str(t)+v)
        inv_str = ' + '.join(inv_str)
        inv_str += ' == 0'
    else:
        inv_str = str(inv)
    return inv_str


def run_ablation(pnames, abl_tests=1, random_seed=False):
    print('Performing Ablation experiment')

    if random_seed:
        seed = random.randint(1, 10_000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print('seed:', seed)
    else:
        if abl_tests > 1:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

    if abl_tests == 1:
        random_seed = False
    else:
        random_seed = True

    summary = defaultdict(list)
    results = defaultdict(list)
    stamp = str(datetime.datetime.now().strftime('%m-%dT%H-%M-%S'))

    abl_names = ['data_norm', 'weight_reg', 'dropout', 'fractional_sampling']
    abl_settings = np.invert(np.diag(np.ones(4, dtype=np.bool))).tolist()

    start_time = time.time()
    total_tests = len(pnames)*4*abl_tests
    cur_test = 1
    for p in pnames:
        print('Running:', p)
        problem = PROBLEMS[p]
        summary['problem'].append(p)
        for abl_name, abl_setting in zip(abl_names, abl_settings):
            print('Testing', abl_name)
            successes = 0
            for test in range(abl_tests):
                data_norm, weight_reg, dropout, fractional_sampling = abl_setting

                elapsed = time.time()-start_time
                remaining = 0.0
                if cur_test > 1:
                    avg = elapsed/(cur_test-1)
                    remaining = avg * (total_tests - (cur_test-1))
                print('Test {}/{}, total elapsed time {:0.1f}s, est. remaining {:0.1f}s'.format(cur_test, total_tests, elapsed, remaining))
                cur_test += 1
                
                loop_successes = 0
                for i in range(len(problem)):
                    loop_idx = i+1
                    print(p, 'loop', loop_idx)
                    
                    problem[i].normalization = data_norm
                    problem[i].weight_reg = weight_reg

                    old_dropout = problem[i].dropout
                    if not dropout:
                        problem[i].dropout = 0.0
                    problem[i].fractional_sampling = fractional_sampling

                    res = problem[i].run(random_seed=random_seed)
                    invariant, z3_vars = compose_invariant(res[0], res[1], res[2], res[3], p)

                    check_res, full_invariant, doc_invariant = check_nla_invariant(invariant, p, loop_idx)
                    
                    loop_successes += check_res
                    print('Correct Invariant Learned?:', str(check_res).upper())

                    results['problem'].append(p)
                    results['loop'].append(i)
                    results['solved'].append(check_res)
                    results['ablation'].append(abl_name)

                    problem[i].dropout = old_dropout
                
                # count success if model learns all loops
                successes += loop_successes == len(problem)

            summary[abl_name].append(successes/(abl_tests))
        dfsummary = pd.DataFrame(summary)
        dfresults = pd.DataFrame(results)

        dfsummary.to_csv('ablation.summary.'+stamp+'.csv')
        dfresults.to_csv('ablation.results.'+stamp+'.csv')

    print('Ablation Results:')
    print(dfsummary)
    print('results saved to ablation.summary.'+stamp+'.csv and ablation.results.'+stamp+'.csv')



def run_stability(random_seed=False):
    print('Performing Stability experiment')
    ntests = 20
    pnames = ['ps2', 'ps3']
    stamp = str(datetime.datetime.now().strftime('%m-%dT%H-%M-%S'))

    if not random_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    # seed entire experiment instead of individual problems
    random_seed = True

    summary = defaultdict(list)

    for p in pnames:
        print('Running:', p)
        problem = PROBLEMS[p]
        summary['problem'].append(p)
        successes = 0
        for i in range(len(problem)):
            loop_idx = i+1
            print(p, 'loop', loop_idx)

            loop_successes = 0
            for test in range(ntests):
                res = problem[i].run(random_seed=random_seed)
                invariant, z3_vars = compose_invariant(res[0], res[1], res[2], res[3], p)
                check_res, full_invariant, doc_invariant = check_nla_invariant(invariant, p, loop_idx)
                successes += check_res
                
                print('Correct Invariant Learned?:', str(check_res).upper())

        summary['stability'].append(successes/ntests)
        dfsummary = pd.DataFrame(summary)

        dfsummary.to_csv('run_nla.stability.summary.csv')

    for i in [1, 11]:
        print('Running:', 'code2inv'+str(i))
        successes = 0
        for test in range(ntests):
            solved, I = gcln_infer(i)
            successes += solved
        summary['problem'].append('code2inv'+str(i))
        summary['stability'].append(successes/ntests)
        dfsummary = pd.DataFrame(summary)
        dfsummary.to_csv('stability.summary.'+stamp+'.csv')

    print('Stability Results:')
    print(dfsummary)
    print('results saved to stability.summary.'+stamp+'.csv')


if __name__ == '__main__':

    ps = PROBLEMS

    valid_options = ["problem=", "data_norm=", "weight_reg=", "dropout=", "fractional_sampling=", "ablation", "stability", "random_seed", "ntests="]

    normalization, regularization, dropout, fractional_sampling = True, True, True, True
    try:
        args = sys.argv
        opts, args = getopt.getopt(args[1:], '', valid_options)
    except getopt.GetoptError:
        print('Invalid command-line argument')
        sys.exit(2)

    pnames = []
    ablation = False
    stability = False
    random_seed = False
    ntests = 1
    for opt, arg in opts:
        if opt == '--problem':
            pnames.append(arg)
        elif opt == '--data_norm':
            if arg in ('false', 'False'):
                normalization = False
        elif opt == '--weight_reg':
            if arg in ('false', 'False'):
                regularization = False
        elif opt == "--dropout":
            if arg in ('false', 'False'):
                dropout = False
        elif opt == '--fractional_sampling':
            if arg in ('false', 'False'):
                fractional_sampling = False
        elif opt == '--ablation':
            ablation = True
        elif opt == '--stability':
            stability = True
        elif opt == '--random_seed':
            random_seed = True
        elif opt == '--ntests':
            ntests = int(arg)

    if len(pnames) == 0:
        pnames = list(ps.keys())

    if ablation:
        run_ablation(pnames, abl_tests=ntests, random_seed=random_seed)
        sys.exit(0)

    elif stability:
        run_stability(random_seed=random_seed)
        sys.exit(0)

    summary = defaultdict(list)
    details = defaultdict(list)
    stamp = str(datetime.datetime.now().strftime('%m-%dT%H-%M-%S'))
    total_time = 0
    total_tests = 0
    for p in pnames:
        print('Running:', p)
        problem = ps[p]
        problem_results = []
        for i in range(len(problem)): 
        # for i in [1]: 
            # try:
            loop_idx = i + 1
            print(p, 'loop', loop_idx)
            if normalization is False:
                problem[i].normalization = False
            if regularization is False:
                problem[i].weight_reg = False
            if dropout is False:
                problem[i].dropout = 0.0
            if fractional_sampling is False:
                problem[i].fractional_sampling = False
            
            for test in range(ntests):
                start_time = time.time()
                res = problem[i].run(random_seed=random_seed)
                elapsed = time.time() - start_time
                total_time += elapsed
                total_tests += 1
                invariant, z3_vars = compose_invariant(res[0], res[1], res[2], res[3], p)
                check_res, full_invariant, doc_invariant = check_nla_invariant(invariant, p, loop_idx, ctx_simplify=True)

                print('Runtime: {:0.1f}s'.format(elapsed))
                print('Learned Invariant:')
                print(str(full_invariant))
                print('Documented Invariant:')
                print(str(doc_invariant))
                print('Correct Invariant Learned?:', str(check_res).upper())

                details['problem'].append(p)
                details['loop_idx'].append(i)
                details['solved'].append(check_res)
                details['learned_inv'].append(str(full_invariant))
                details['documented_inv'].append(str(doc_invariant))
                if len(pnames) > 1:
                    dfdetails = pd.DataFrame(details)
                    dfdetails.to_csv('run_nla.results.'+stamp+'.csv')

                problem_results.append(check_res)

        summary['problem'].append(p)
        summary['solved'].append(np.all(problem_results))

        # if running multiple problems save summary:
        if len(pnames) > 1:
            dfsum = pd.DataFrame(summary)
            dfsum.to_csv('run_nla.summary.'+stamp+'.csv')
    
    if len(pnames) > 1:
        print('Summary:')
        print(pd.DataFrame(summary))
        print('Avg Runtime: {:0.1f}'.format(total_time/total_tests))
        print('results saved to run_nla.summary.'+stamp+'.csv and run_nla.results.'+stamp+'.csv')

