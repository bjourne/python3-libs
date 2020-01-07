import itertools
import sys

def alternatives(ch, repls):
    key = ch.lower()
    if key in repls:
        return [ch, repls[key]]
    return [ch]

def main(argv):
    if len(argv) != 2:
        print(f"USAGE: {argv[0]} [word]")
        return

    # Populate the known replacements
    replacements = {'a' : '4', 'e' : '3', 'i' : '1',
                    'm' : '/v\\', 'o' : '0', 'r' : '2'}

    s = [alternatives(ch, replacements) for ch in argv[1]]
    for it in itertools.product(*s):
        print(''.join(it))

if __name__ == '__main__':
    main(sys.argv)
