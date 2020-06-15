#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[69]:


import re
from subprocess import run
import json


OPERATORS = set(['+', '-', '*', '/', '(', ')', '@', '<', '#', '>', '!', '='])
PRIORITY = {'+':1, '-':1, '*':2, '/':2}
def infix_to_prefix(formula):
    op_stack = []
    exp_stack = []
    for ch in formula:
        if not ch in OPERATORS:
            exp_stack.append(ch)
        elif ch == '(':
            op_stack.append(ch)
        elif ch == ')':
            while op_stack[-1] != '(':
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append( " ".join(["(",op,b,a,")"]) )
            op_stack.pop() # pop '('
        else:
            while op_stack and op_stack[-1] != '(' and PRIORITY[ch] <= PRIORITY[op_stack[-1]]:
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append( " ".join([op,b,a]) )
            op_stack.append(ch)

    
    # leftover
    while op_stack:
        op = op_stack.pop()
        a = exp_stack.pop()
        b = exp_stack.pop()
        exp_stack.append(  " ".join([op,b,a] ))
    return exp_stack[-1]

def clean_up (line, paren=False):
    left = "" 
    right = ""
    if paren:
        left = "( "
        right = " )"
    
    clean = line.split('(')[1:]
    clean = left.join(clean)
    clean = clean.split(')')[:-1] 
    clean = right.join(clean)
    
    return clean.strip()
    
def full_prefix(line):
    clean = line
    if "(" in line:    
        clean = clean_up(line, paren=True)
    

    encode = re.sub(r'<=', '@', clean)
    encode = re.sub(r'>=', '#', encode)
    encode = re.sub(r'!=', '!', encode)
    encode = re.sub(r'==', '=', encode)

    prefix = infix_to_prefix(encode.split())
    decode = re.sub(r'@', '<=', prefix)
    decode = re.sub(r'#', '>=', decode)
    decode = re.sub(r'!', '!=', decode)

    
    if decode.strip()[0] == "(":
        decode = clean_up(decode,paren=True)

    return decode

def op_conversion(l):
    if '==' in l:
        l = re.sub(r'==', '=', l)
    if "!=" in l:
        l = "not (" + re.sub(r'!=', '=', l) + ")"
    return l

def to_prefix(l,line):
    if len(l.split()) != 3:
        out = full_prefix(line)
        #print("WARNING: MORE THAN 3 TOKENS",out)
    else:
        out = full_prefix(l)
    return out
    
    

# with open('../code2inv/our_preprocess/_loop_cond.txt', 'wt') as output_file:
# for i in range(1, 134):
#
def parse_conditions():
    run(['mkdir', '-p', '../benchmarks/code2inv/conditions'])
    for i in range(1,134):
        with open('../benchmarks/code2inv/c/' + str(i) + '.c', 'rt') as input_file:
            preconditions = []
            predicate = None
    
            post_conditions = {'ifs':[], 'assert':None}
            
            before_loop = True
            post_condition = False
            ifs = []     
            
            for line in input_file:
    
    #             print(line.strip())
                
        
                if "while" in line:
                    before_loop = False
                    if ("unknown" in line):
    #                     print(str(i) + '\t' )
                        continue
                    l = clean_up(line)
                    l = to_prefix(l,line)
                    
                    #print (str(i) + '\t' + (clean) + '\n')
                    
                    l = op_conversion(l)
    
    #                 print (str(i) + '\t' + decode)
                    predicate = "( " + l + " )"
                    
                if before_loop:
    #                 print(re.split('\(|\)', line))
                    if "assume" in line:
                        l = clean_up(line)
                        
                        l = to_prefix(l,line)
                        l = op_conversion(l)
    #                     print('(' + l + ')')
                        preconditions.append('(' + l + ')')
                    elif '=' in line:
                        
                        line = line.strip()
                        line = line.strip(';')
                        l = line.strip('()')
                        
                        if l.split()[0] == 'int':
                            
                            l = l.split()
                            l = " ".join(l[1:])
                        
                        l = to_prefix(l,line)
                        l = op_conversion(l)
                        preconditions.append('(' + l + ')')
                
                if "post-condition" in line:
                    post_condition = True
                    
        
                if '//' in line:
                    continue
                if post_condition and "if" in line                    and post_condition == True:
                    l = clean_up(line)
                    l = to_prefix(l,line)
                    
                    l = op_conversion(l)
                    ifs.append("( not ("+ l +"))")
    #                 print (ifs)
                if 'assert' in line:
                    l = clean_up(line)              
                    l = to_prefix(l,line)
                    l = " ".join(l.split())
                    l = op_conversion(l)
    #                 out = "or "
    #                 for if_cond in ifs:
    #                     out = out + if_cond
    #                 out = out + "(" + l + ")"
                    post_conditions['ifs'] = ifs
                    post_conditions['assert'] = "( " + l + " )"
                    
            conditions = {'preconds': preconditions, 'predicate': predicate,
             'postcondition': post_conditions}
            #print (conditions)
            with open('../benchmarks/code2inv/conditions/' + str(i) + '.json', 'w') as f:
                json.dump(conditions, f)
    
    

# In[72]:



import subprocess
from shutil import copyfile
import json

def fast_check(inv,i):
    with open ('./tmp.smt', "w") as f:
        f.write(inv)
    p = subprocess.run(['./check.sh', './tmp.smt', '../benchmarks/code2inv/smt2/' + str(i) + ".c"], stdout=subprocess.PIPE)
    return p.stdout.decode().count("unsat")
    
def fast_checker(i):
    I = None
    solved = False
    with open('../benchmarks/code2inv/conditions/' + str(i) + '.json', 'r') as f:
        condition = json.load(f)
    
    collection = []
    collection = collection + (condition['preconds'])
    collection.append(condition['predicate'])
    collection.append(condition['postcondition']['assert'])
    collection = collection + (condition['postcondition']['ifs'])
    if condition['predicate']:
        collection.append( "(or " + condition['predicate'] + " " + condition['postcondition']['assert'] + ")")
    collection = [i for i in collection if i is not None]
    ands = [ "(and " + i + " " + j + " )" for i in collection for j in collection]
    ors = [ "(or " + i + " " + j + " )" for i in collection for j in collection]
    collection = collection + ands + ors
    for inv in collection:
        if (fast_check(inv,i) == 3):
            solved = True
            I = inv
            break
    if solved:
        #copyfile ('./tmp.smt', '../../results/basic_filter/' + str(i) + ".c.inv.smt")
        subprocess.run(['mkdir', '-p', './tmp_result'])
        copyfile ('./tmp.smt', './tmp_result/' + str(i) + ".c.inv.smt")
    return solved, I


if __name__ == '__main__':
    for i in range(1, 134):
        fast_checker(i)
