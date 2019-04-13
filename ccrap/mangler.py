# Names and name mangling.
from html.entities import codepoint2name
from string import ascii_letters, digits

C_SAFE = ascii_letters + digits + '_'

NAME_2_MANGLED = {}
MANGLED_2_NAME = {}

TRANS = {
    '=' : '_eq',
    '?' : '_qm',
    '!' : '_em',
    '-' : '_sub',
    '+' : '_add',
    '*' : '_mul',
    '/' : '_div'
}

def encode(ch):
    if ch in C_SAFE:
        return ch
    o = ord(ch)
    if o in codepoint2name:
        return codepoint2name[o]
    if ch in TRANS:
        return TRANS[ch]
    raise Exception("Couldn't encode `%s`!" % ch)

def mangle_new(name):
    return 'ccrap_%s' % ''.join(encode(ch) for ch in name)

def mangle(name):
    if name not in NAME_2_MANGLED:
        new_name = mangle_new(name)
        i = 0
        while True:
            if new_name not in MANGLED_2_NAME:
                break
            new_name = mangle_new('%s_%02d' % (name, i))
        NAME_2_MANGLED[name] = new_name
        MANGLED_2_NAME[new_name] = name
    return NAME_2_MANGLED[name]

COUNTERS = {'var' : 0, 'quot' : 0}
def hit_counter(counter):
    val = COUNTERS[counter]
    COUNTERS[counter] += 1
    return val

def generate_name(type):
    return '%s_%s' % (type, hit_counter(type))

if __name__ == '__main__':
    print(mangle('long_name'))
