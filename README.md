# Gated CLN Loop Invariant Learning

## Setup Instructions (tested on ubuntu 18.04):
1. Install miniconda for pyton 3.7: https://docs.conda.io/en/latest/miniconda.html
2. Install pytorch `conda install pytorch cpuonly -c pytorch`
  See https://pytorch.org/get-started/locally/ for other pytorch install options
3. Install other dependencies
```bash
conda install pandas matplotlib
pip install tqdm
pip install z3-solver sklearn
```


## Running:
#### NLA benchmark (Polynomial Invariants)
```bash
cd gcln_model;
python run_nla.py
```


The script will run through each problem in the nla benchmark and print learned invariant and check result. 
It will also print a table replicating table 2 in the paper, and save the results in `run_nla.summary.<timestamp>.csv`. 
The expected runtime is 30 minutes.

The script will print out both the learned and documented invariant, along with the validity check result. Since a problem may have many possible invariants, we evaluate the learned invariant by checking whether  it is sufficient to prove the documented invariant using the z3 solver.

You may also run individual problems, for example to run 'ps2':

```bash
cd gcln_model;
python run_nla.py --problem=ps2
```

A fixed random seed is used to have reproducible results. To run with a randomly selected seed use the `--random_seed` flag.


#### Code2Inv benchmark (Linear Invariants)
```bash
cd gcln_model;
python run_code2inv.py
```

The script will run through each problem in the code2inv benchmark and print out the learned invariants, whether it passes the benchmark check, and a summary of all results.
This experiment is expected to complete within 15 minutes.

You may also run individual problems. For example, to run problem 5:
```bash
python run_code2inv.py 5
```

#### Ablation study

To replicate Table 3 in the paper, run nla with the following configurations:

Full method: use results from 
```bash
python run_nla.py
```

To perform the ablation to replicate the results in the paper run
```bash
python run_nla.py --ablation
```

This will run the ablation on each problem and print the ablation portion of table 3 (the full method column results are the same as running `run_nla.py` without any options) It will also save the results in `ablation.summary.<timestamp>.csv`. 
The expected runtime is within 2 hours. 
Note that z3 timeouts are treated as failures, so results may differ if the experiment is run on a poorly specced or heavily loaded system.
 
The ablation may also be run on individual problems. 
To run with multiple tests per problem use the `--ntests=` flag.
Thus to run the ablation test on `ps2` and `ps3`, one may run:
```bash
python run_nla.py --ablation --ntests=5 --problem=ps2 --problem=ps3
```

This experiment is also seeded at the beginning to have reproducible results, to run with randomly selected seeds use the `--random_seed` flag. For example, the previous command to run ablation on `ps2` and `ps3` can be made random with:
```bash
python run_nla.py --ablation --ntests=5 --problem=ps2 --problem=ps3 --random_seed
```


#### Stability Test

To replicate the "G-CLN" column in Table 4 in the paper, run
```bash
python run_nla.py --stability
```

The results are saved in `stability.summary.<timestamp>.csv`. The two handcrafted examples from the CLN authors [30] are not included in this artifact. Note that z3 timeouts are treated as failures, so stability may be lower if the experiment is run on a poorly specced or heavily loaded system. 
This experiment is expected to complete within 30 minutes.




## Artifact Structure

The main directories and files in this artifacts are described below.

- benchmarks/
  - nla/: the non-linear loop invariant benchmark [21]
  - code2inv/: the linear loop invariant benchmark [35]
- gcln_model/
  - gcln_model.py: the equality learning model of G-CLN 
  - ineq_solver.py: the inequality learning model of G-CLN
  - poly_template_gen.py: data preparation and preprocessing before model fitting
  - inv_postprocessing.py: invariant extraction, filtering and cleaning after model fitting


