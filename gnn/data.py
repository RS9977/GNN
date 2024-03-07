import os
import re
import glob
import numpy as np

import torch
from torch_geometric.data import Data

from tokenizer import process_bb_tokenizer

'''
Take from Reza's colab notebook
'''


def live_feat(filename, function_num):
    # The file name should be dumped using -fdump-tree-ssa-gimple. It's the
    # 021 tree.
    lines = []
    check_file = 0
    if filename =='data/multiwlinear.c.023t.ssa.c':
        check_file = 1
    with open(filename, "r") as file:
        for line in file:
            lines.append(line.strip())

    ##########################################################################
    # Create the list of using variables.

    function_name = 0  # Here I just enumerate
    initial_part = 0
    var_dict_temp = {}  # It's local for any function and has the temperory vars
    var_dict_static = {}  # It's local for any function and has the args and defined vars
    var_all = {}  # Key's are corresponded to the funcation name and values are the var_dict_temp of each function
    fun_def = 0  # check for __GIMPLE to see if we are at the beginning of a function

    for line in lines:
        words = line.split()

        if fun_def:  # since we know the immidiate line after the __GIMPLE is the argument lines of the function
            fun_def = 0
            args = line.split(sep=',')  # first split it based on ','
            for arg in args:
                arg_list = arg.split()
                if '(' not in arg:  # check if it is the first arguman and has the function name and other stuff in it
                    if ')' not in arg:  # check if it is the last arguman and has the last ')'
                        var_dict_static[arg_list[-1]
                                        ] = arg[1:-(len(arg_list[-1]) + 1)]
                    else:
                        var_dict_static[arg_list[-1][0:-1]
                                        ] = arg[1:-(len(arg_list[-1]) + 1)]
                else:
                    par_ind = arg.find('(')
                    if ')' not in arg:
                        var_dict_static[arg_list[-1]
                                        ] = arg[par_ind + 1:-(len(arg_list[-1]) + 1)]
                    else:
                        if '(' not in arg_list[-1]:
                            var_dict_static[arg_list[-1][0:-1]
                                            ] = arg[par_ind + 1:-(len(arg_list[-1]) + 1)]

        for word in words:
            if '__GIMPLE' in word:  # check for the __GIMPLE that is a distinctive annotation for the beginning of the function
                fun_def = 1

            if '{ ' == word:  # check to see if we are in the function scope.
                initial_part = 1  # since in the SSA all the initializations are at the beginning
            if '}' == word:  # check to see if we are out of the function scope.
                var_all_local = {}  # to have temp and static vars in dict
                var_all_local['static'] = var_dict_static
                var_all_local['temp'] = var_dict_temp
                # updating var_all for values in the parsed function
                var_all[function_name] = var_all_local
                var_dict_temp = {}  # clear the local var_dict_temp
                var_dict_static = {}  # clear the local var_dict_static
                function_name += 1  # increase the index of function
            if '__BB' in word:
                initial_part = 0  # check the end of initialization

        if initial_part and len(words) > 1:
            # having '_' at the beginning indicates it's temperoray and made by
            # the compiler
            if words[-1][0] == '_':
                var_dict_temp[words[-1][0:-1]] = line[0:-(len(words[-1]) + 1)]
            else:
                var_dict_static[words[-1][0:-1]
                                ] = line[0:-(len(words[-1]) + 1)]

    ##########################################################################
    # Create the adjacent matrix for BBs
    fun_def = 0  # check for __GIMPLE to see if we are at the beginning of a function
    function_scope = 0  # to check if we are inside a scope
    function_name = 0  # Here I just enumerate
    BB_name = 0  # to have the name of current BB. It's only valid if BB_scope=1
    BB_ind = -1  # to store a numeric value for BBs.
    BB_scope = 0  # to check if we are inside a scope
    BB_first_line = 0  # to store the BBs name and info
    # it has the 0-indexed vertex as key and BBs name in val for each function
    BB_local_dict = {}
    # it has the 0-indexed vertex as key and BBs name in val for all functoins
    BB_global_dict = {}
    # it has the 0-indexed vertex as val and BBs name in key for each function
    BB_rev_local_dict = {}
    # it has the 0-indexed vertex as val and BBs name in key for all functoins
    BB_rev_global_dict = {}
    BB_adjacent_mat = []  # it's for all the BBs inside a function
    BB_local_adj = []  # it's local to each BB in a function
    BB_adjacent_mat_global = {}  # it's for all of the funcitons
    BB_all_val = []  # have all the variable use in a function
    BB_all_val_global = {}  # have entire variables

    BB_instr_global = {}  # SRA
    BB_instr = []
    BB_contents = []
    for line in lines:
        words = line.split()

        if fun_def:  # since we know the immidiate line after the __GIMPLE is the argument lines of the function
            fun_def = 0

        if '__GIMPLE' in line:  # check for the __GIMPLE that is a distinctive annotation for the beginning of the function
            fun_def = 1

        
        # check to see if we are out of the function scope.
        if '}' in line and len(line) < 2:
            BB_instr_global[function_name] = BB_instr
            BB_instr = []
            function_scope = 0
            BB_global_dict[function_name] = BB_local_dict
            BB_local_dict = {}
            BB_rev_global_dict[function_name] = BB_rev_local_dict
            BB_rev_local_dict = {}
            if len(BB_local_adj) > 0:
                BB_adjacent_mat.append(BB_local_adj)
            BB_adjacent_mat_global[function_name] = BB_adjacent_mat
            BB_local_adj = []
            BB_adjacent_mat = []
            BB_ind = -1
            function_name += 1  # increase the index of function

        # check to see if we are in the function scope.
        if '{' in line and len(line) < 2:
            function_scope = 1  # check if we are inside of the function scope

        if function_scope == 1:
            if line[0:4] == '__BB':
                BB_scope = 1  # check that we are inside a BB
                BB_first_line = 1
                if len(BB_contents) > 0:
                    if line[4:7] != '(2)':
                        BB_instr.append(BB_contents)
                BB_contents = []

            if line != '':
                BB_contents.append(line)
            
            if BB_first_line:  # store BBs data
                if BB_ind >= 0:  # not for the first one
                    if len(BB_local_adj) > 0:
                        BB_adjacent_mat.append(BB_local_adj)
                    else:
                        BB_ind -= 1
                BB_local_adj = []
                BB_first_line = 0  # reset the first line flag for BBs
                start = line.find('(') + 1
                if line.find(',') > 0:  # check if there is other info
                    end = line.find(',')
                else:
                    end = line.find(')')
                BB_name = line[start: end]
                BB_ind += 1
                # update the local dict for BBs' name
                BB_local_dict[BB_ind] = int(BB_name)
                BB_rev_local_dict[int(BB_name)] = BB_ind

            if BB_scope:
                if ' =' in line and 'if' not in line:  # Check if there is an assignment. This can be consider as the start of the livenss analysis since the tree is in SSA form

                    for word in words:
                        var = ''
                        vars = set()
                        if '_' in word and '__' not in word:
                            for char in word:
                                asci = ord(char)
                                if (asci <= 90 and asci >= 65) or (asci <= 57 and asci >= 48) or (asci <= 122 and asci >= 97) or char == '_':
                                    var += char
                                else:
                                    if len(var) > 0:
                                        vars.add(var)
                                        var = ''
                            vars.add(var)


                if 'goto' in line and '//' not in line:
                    start = line.find('B') + 2
                    BB_local_adj.append(int(line[start:-1]))
                if 'return' in line and '[return' not in line and '//' not in line:
                    BB_local_adj.append(-1)
                    if len(BB_contents) > 0:
                        BB_instr.append(BB_contents)
                    BB_contents = []
        

    ##########################################################################
    #print(var_all)
    BB_ind = 0
    adj_list = []
    for BB_local_adj in BB_adjacent_mat_global[function_num]:
        for adj in BB_local_adj:
            if adj > 0:
                adj_list.append(
                    [BB_global_dict[function_num][BB_ind] - 2, adj - 2])
        BB_ind += 1

    return [adj_list, BB_instr_global[function_num]]

# ----------------------

# data = read_graphs_for_rnn('data', live_feat, function_num=0)
# read_graphs_for_rnn calls process_bb
#
#

def extract_a_op_b(input_string):
    

    # Find and replace expressions within parentheses with a placeholder
    placeholder = '__PLACEHOLDER__'
    flag_par = 0
    while '(' in input_string and ')' in input_string:
        flag_par = 1
        # Find the innermost expression within parentheses
        inner_expr = re.search(r'\([^)]*\)', input_string).group(0)
        
        # Replace the inner expression with a placeholder
        input_string = input_string.replace(inner_expr, placeholder)

    # Define the original pattern
    original_pattern = r"([-+]?\b\d+(\.\d*)?(e[+-]?\d+)?\b|\b\w+\b|\[\s*\d+(\s*,\s*\d+)*\s*\])\s*([+\-*/])\s*([-+]?\b\d+(\.\d*)?(e[+-]?\d+)?\b|\b\w+\b|\[\s*\d+(\s*,\s*\d+)*\s*\])"

    # Find the first match in the modified input string
    match = re.search(original_pattern, input_string)

    if match:
        # Extract a, op, b from the match
        a = match.group(1)
        op = match.group(5)
        b = match.group(6)

        # Replace the placeholder back with the original expression
        if flag_par:
            a = a.replace(placeholder, inner_expr)
            b = b.replace(placeholder, inner_expr)

        return op
    else:
        return None



def process_bb(bb_list):
    ops = []
    for bb in bb_list:
        ins_list = []
        for ins in bb:
            #ins.find('=') > -1
            if '=' in ins and '==' not in ins:
                ins_list.append(ins.split('=')[1].strip())
        ops.append(ins_list)
    expr = '[a-z]*_[0-9]{1,3}[(D)]*'
    ops_clean = []
    for ins_list in ops:  # loop over bbs
        ins_clean_list = []

        for line in ins_list:  # loop over ins in a bb
            # regex matching
            sym = extract_a_op_b(line)
            found = False
            if sym in ['+', '-', '*', '/']:
                
                line = sym
                found = True
            else:
                if m := re.findall(expr, line):                
                    for k in m:
                        line = line.replace(k, "ARG")

                        # CUSTOM PROCESSING PIPELINE HERE
                        # going to be very hacky for now
                        line = line.replace("ARG", "")
                        line = line.replace("(long unsigned int)", "")

                        line = line.strip().rstrip(";").strip()

            
                if line.find('__MEM') > -1:
                    line = 'MR'
                    found = True

                if line.find("PHI") > -1:
                    line = 'PHI'
                    found = True

                if re.findall('^[0-9]*;', line):
                    line = ''
 
            # if len(line): ins_clean_list.append(line)
            if found:
                ins_clean_list.append(line)

        if len(ins_clean_list):
            ops_clean.append(ins_clean_list)
        else:
            ops_clean.append(['BLANK'])

    return ops, ops_clean


def read_graphs_for_rnn(folder, func=live_feat, **kwargs):
    files = glob.glob(f'{folder}{os.sep}*.ssa.c')
    #print(files[6])
    #graphs = [func(f, **kwargs) for f in files]
    graphs = []
    for f in files:
        for i in range(5):
            try:
                graph = func(f, i)
                if(len(graph[1])>1):
                    graphs.append(graph)
            except:
                break
    data = []
    for g in graphs:
        edge_index = torch.tensor(list(g[0])).t().contiguous()

        bb_list = g[1]  # list of bb instructions

        #ops, ops_clean = process_bb(bb_list)
        #data.append((g[0], ops, ops_clean))
        #print(bb_list)
        tokens = process_bb_tokenizer(bb_list)
        data.append((g[0], tokens))
    #assert len(data) == len(files)

    return data


def map_ins_to_idx(data, word_to_idx):
    data_emb = []
    for g in data:
        #edge_index, _, bb_ins = g
        edge_index, bb_ins = g
        bb_ins_idx = []
        for bb in bb_ins:
            bb_ins_idx.append(torch.tensor([word_to_idx[k] for k in bb]))

        # bb_ins_idx = pad_sequence(bb_ins_idx, padding_value=len(word_to_idx)+1)
        if len(edge_index) != 0:
            edge_index = torch.tensor(edge_index).T
        else:
            edge_index = torch.tensor([[0],[0]])
        data_emb.append((edge_index, bb_ins_idx))

    return data_emb


def prepare_data_vocab(folder,
                       func=live_feat,
                       **kwargs):

    data = read_graphs_for_rnn(folder, func=func, **kwargs)
    
    vocab = []
    #for app in [d[2] for d in data]:
    for app in [d[1] for d in data]:
        for bb in app:
            for line in bb:
                vocab.append(line)
    
    vocab = np.unique(vocab)
    
    word_to_idx = dict(zip(vocab, np.arange(len(vocab))))
    idx_to_word = dict(zip(np.arange(len(vocab)), vocab))

    for w in word_to_idx:
        assert idx_to_word[word_to_idx[w]] == w

    data_emb = map_ins_to_idx(data, word_to_idx)

    return word_to_idx, idx_to_word, data, data_emb

# ----------------------

# These are functions when not generating embeddings
# based on an RNN but when using hand-generated features
#

# WARNING: no sanity checks have been done for the basic blocks


def graph_to_data(G, label):
    edge_index = torch.tensor(list(G[1])).t().contiguous()

    x = torch.tensor(G[0])

    y = torch.tensor([label])

    return Data(x=x, edge_index=edge_index, y=y)


def read_dataset(folder):
    g = read_graphs(folder)

    g = [graph_to_data(x, idx) for idx, x in enumerate(g)]

    return g


def read_graphs(folder, func=live_feat, **kwargs):
    files = glob.glob(f'{folder}{os.sep}*.ssa.c')

    #graphs = [func(f, **kwargs) for f in files]
    graphs = []
    for f in files:
        for i in range(5):
            try:
                graph = func(f, i)
                if(len(graph[1])>1):
                    graphs.append(graph)
            except:
                break

    return graphs
