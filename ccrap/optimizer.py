def optimize_dip_check(seq, el):
    return el == ('sym', 'dip') and seq[-1][0] == 'quot'

def optimize_dip(seq):
    _, dip_seq = seq.pop()
    seq.append(('dip', dip_seq))

def optimize_if_check(seq, el):
    return el == ('sym', 'if') and seq[-1][0] == seq[-2][0] == 'quot'

def optimize_if(seq):
    _, false_seq = seq.pop()
    _, true_seq = seq.pop()
    false_seq = optimize_body(false_seq)
    true_seq = optimize_body(true_seq)
    seq.append(('if', [true_seq, false_seq]))

def optimize_for_index_check(seq, el):
    return el == ('sym', 'for-index') and seq[-1][0] == 'quot'

def optimize_for_index(seq):
    _, for_seq = seq.pop()
    seq.append(('for-index', for_seq))

def optimize_el(seq, el):
    if optimize_if_check(seq, el):
        optimize_if(seq)
    elif optimize_for_index_check(seq, el):
        optimize_for_index(seq)
    elif optimize_dip_check(seq, el):
        optimize_dip(seq)
    else:
        seq.append(el)

def optimize_body(seq):
    new_seq = []
    for el in seq:
        optimize_el(new_seq, el)
    return new_seq

def optimize_def(df):
    type, node = df
    if type != 'def':
        return df
    name, effect, seq = node
    new_seq = optimize_body(seq)
    return type, (name, effect, new_seq)

def optimize(defs):
    return [optimize_def(df) for df in defs]
