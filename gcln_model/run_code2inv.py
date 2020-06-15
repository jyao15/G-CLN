from gcln_code2inv import gcln_infer
from condition_parser import parse_conditions, fast_checker
import time
import sys, os
from z3 import Real, And
import numpy as np


def run_code2inv_problem(i):
    verbose=False
    solved=False
    if i in [26, 27, 31, 32, 61, 62, 72, 75, 106]:
        print(i,'theoretically unsolvable')
        return True, 0

    start_time = time.time()
    attempts = 0
    first_time = True
    max_epoch = 2000
    I = None
    while (solved == False):
        attempts += 1
        if attempts > 10:
            # print('failed', i)
            # return False
            break

        if first_time:
            solved, I = fast_checker(i)
        if solved:
            break
        else:
            non_loop_invariant = None
            if i in [110, 111, 112, 113]:
                non_loop_invariant = And(Real('sn') == 0, Real('i') == 1, Real('n') < 0)
            elif i in [118, 119, 122, 123]:
                non_loop_invariant = And(Real('sn') == 0, Real('i') == 1, Real('size') < 0)

            solved, I = gcln_infer(i, max_epoch=max_epoch, 
                                    non_loop_invariant=non_loop_invariant) 

            first_time = False
            max_epoch += 1000

    end_time = time.time()
    runtime = end_time - start_time

    print('Problem number:', i, 'solved?',solved, 'time:', runtime)
    if I is not None:
        print(I)
    return solved, runtime

if __name__=='__main__':
    if not os.path.isdir('../benchmarks/code2inv/conditions'):
        print('preprocessing source files...')
        parse_conditions()
    if len(sys.argv) > 1:
        print('running problem', sys.argv[1])
        problem = int(sys.argv[1])
        run_code2inv_problem(problem)
    else:
        print('running entire code2inv benchmark, use python run_code2inv.py <problem_number> to run single problem')
        total_solved = 0
        total = 0
        unsolvable = 0
        runtimes = []
        for i in range(1, 134):
            solved, runtime = run_code2inv_problem(i)
            if i in [26, 27, 31, 32, 61, 62, 72, 75, 106]:
                unsolvable += 1
            else:
                total += 1
                total_solved += solved
                runtimes.append(runtime)
            print('Solved {}/{} solvable problems, {} theoretically unsolvable problems'.format(total_solved, total, unsolvable))
            print('Avg. Runtime: {:0.1f}s, Max Runtime: {:0.1f}s'.format(np.mean(runtimes), np.max(runtimes)))
        print()
        print('Summary:')
        print('Solved {}/{} solvable problems, {} theoretically unsolvable problems'.format(total_solved, total, unsolvable))
        print('Avg. Runtime: {:0.1f}s, Max Runtime: {:0.1f}s'.format(np.mean(runtimes), np.max(runtimes)))


