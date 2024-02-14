import re

def preprocess_instruction(instruction):
    # Replace variables followed by '->' and another variable with TMP_POINTER_MEMBER
    instruction = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*(?<!__MEM)(?<!__BB)(?<!__PHI))->([a-zA-Z_][a-zA-Z0-9_]*(?<!__MEM)(?<!__BB)(?<!__PHI))(?![^<>]*>)', 'TMP_POINTER_MEMBER', instruction)
    # Replace function names
    instruction = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?<!__MEM)(?<!__BB)(?<!__PHI)(?![^<>]*>)(?=\);)', 'TMP_FUNCTION_NAME', instruction)
    # Replace other variable patterns
    instruction = re.sub(r'\b([a-zA-Z_]+[0-9_]+)\b(?<!__MEM)(?<!__BB)(?<!__PHI)(?![^<>]*>)', 'TMP_VAR', instruction)
    
    instruction = re.sub(r'\b_[0-9]+\b', 'TMP_INTERMEDIATE_VAR', instruction)
    
    # Replace numbers
    instruction = re.sub(r'\b[0-9]+(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?)?(?:u|ul|l)?\b', 'TMP_NUMBER', instruction)
    instruction = re.sub(r'\b[a-zA-Z_]+[a-zA-Z0-9_]*\b(?<!__MEM)(?<!__BB)(?<!__PHI)(?<!TMP_VAR)(?<!TMP_FUNCTION_NAME)(?<!TMP_POINTER_MEMBER)(?<!TMP_INTERMEDIATE_VAR)(?<!TMP_NUMBER)(?![^<>]*>)', 'INFUNCTION_VAR', instruction)
    return instruction

def tokenize_gimple_instruction(instruction):
    # Regular expressions to match different types of instructions
    store_pattern = r'__MEM\s+<(\w+)>\s+\((.*?)\)\s+=\s+(.*?);'
    load_pattern = r'([a-zA-Z0-9_]+)\s+=\s+__MEM\s+<(\w+)>\s+\((.*?)\);'
    math_op_pattern = r'([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_.eE+-]+)\s*([+\-*\/])\s*([a-zA-Z0-9_.eE+-]+);'
    function_call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)(?: \((.*?)\))?;'
    assignment_pattern = r'([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_.eE+-]+);'
    phi_pattern = r'([a-zA-Z0-9_]+)\s*=\s* __PHI \((.*?)\);'

    # Match instruction types
    if re.match(store_pattern, instruction):
        return tokenize_store(instruction)
    elif re.match(load_pattern, instruction):
        return tokenize_load(instruction)
    elif re.match(math_op_pattern, instruction):
        return tokenize_math_op(instruction)
    elif re.match(function_call_pattern, instruction):
        return tokenize_function_call(instruction)
    elif re.match(assignment_pattern, instruction):
        return tokenize_assignment(instruction)
    elif re.match(phi_pattern, instruction):
        return tokenize_phi(instruction)
    else:
        return None
    
def tokenize_store(instruction):
    match = re.match(r'__MEM\s+<(\w+)>\s+\((.*?)\)\s+=\s+(.*?);', instruction)
    return {
        'instruction_type': 'store',
        'value_type': match.group(1),
        'address': match.group(2),
        'value': match.group(3),
        'end_of_instruction': ';'
    }

def tokenize_load(instruction):
    match = re.match(r'([a-zA-Z0-9_]+)\s+=\s+__MEM\s+<(\w+)>\s+\((.*?)\);', instruction)
    return {
        'instruction_type': 'load',
        'destination_var': match.group(1),
        'value_type': match.group(2),
        'address': match.group(3),
        'end_of_instruction': ';'
    }

def tokenize_math_op(instruction):
    match = re.match(r'([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_.eE+-]+)\s*([+\-*\/])\s*([a-zA-Z0-9_.eE+-]+);', instruction)
    return {
        'instruction_type': 'math_op',
        'result_var': match.group(1),
        'operand1': match.group(2),
        'operator': match.group(3),
        'operand2': match.group(4),
        'end_of_instruction': ';'
    }

def tokenize_function_call(instruction):
    # Extract function name and arguments
    match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)(?: \((.*?)\))?;', instruction)
    if match:
        function_name = match.group(1)
        arguments = match.group(2)
        if arguments:
            arguments = [arg.strip() for arg in arguments.split(',')]
        out = dict()
        out = {
            'instruction_type': 'function_call',
            'function_name': function_name
        }
        i = 0
        if arguments!= None:
            for arg in arguments:
                i += 1
                arg_key = "argument_" + str(i)
                out[arg_key] = "TMP_FUNCTION_ARG_" + str(i)
        out['end_of_instruction']= ';'
        return out
    else:
        return None

def tokenize_assignment(instruction):
    match = re.match(r'([a-zA-Z0-9_]+)\s*=\s*(.*?);', instruction)
    return {
        'instruction_type': 'assignment',
        'destination_var': match.group(1),
        'source_var': match.group(2),
        'end_of_instruction': ';'
    }

def tokenize_phi(instruction):
    match = re.match(r'([a-zA-Z0-9_]+)\s*=\s* __PHI \((.*?)\);', instruction)
    return {
        'instruction_type': 'phi',
        #'result_var': match.group(1),
        #'arguments': match.group(2),
        'end_of_instruction': ';'
    }

# Example instructions


# Tokenize each instruction
def process_bb_tokenizer(bb_list):
    tokens_list = []
    for bb in bb_list:
        tokens_temp = []
        for ins in bb:

    #for instruction in instructions:
            instruction = ins
            preprocessed_instruction = preprocess_instruction(instruction)
            tokens = tokenize_gimple_instruction(preprocessed_instruction)
            #print(tokens)
            if tokens!=None:
                for key in tokens.keys():
                    tokens_temp.append(tokens[key])
            else:
                tokens_temp.append(';')
        tokens_list.append(tokens_temp)
    return tokens_list



instructions = [[
    "__MEM <double> (root_33(D)) = x_right_25;",
    "x_left_23 = __MEM <double> (x_lower_22(D));",
    "state_20->f_lower = _15;",
    "f_upper_27 = state_20->f_upper;",
    "x_bisect_28 = 2.0e+0 / _1;",
    "GSL_MIN_DBL (abs_lower_12, abs_upper_13);",
    'gsl_error ("function value is not finite", ".//bisection.c", 99, 9);',
    "_2 = _1 + 18446744073709551608ul;",
    "f_bisect = _4;",
    "_16 = __PHI (__BB3: _52, __BB5: _49, __BB7: _46, __BB9: _44, __BB16: _40);",
]]

print(process_bb_tokenizer(instructions))