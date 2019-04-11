from sys import exit

class Definition:
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def emit(self):
        return [
            'top++;',
            'stack[top] = (cell)&%s;' % self.name
        ]

    def emit_body(self):
        lines = [line for node in self.body for line in node.emit()]
        return [' ' * 4 + l for l in lines]

    def __repr__(self):
        cls_name = self.__class__.__name__
        return '%s(%s, %s)' % (cls_name, self.name, self.body)

class Call:
    def __init__(self, name):
        self.name = name

    def emit(self):
        return ['top = %s(stack, top);' % self.name]

class CCall:
    def __init__(self, name, types):
        self.name = name
        self.types = types

    def emit(self):
        fmt = '(%s)stack[top - %d]'
        n_args = len(self.types)
        args = [fmt % (t, n_args - 1 - i) for (i, t) in enumerate(self.types)]
        return [
            '%s(%s);' % (self.name, ', '.join(args)),
            'top -= %d;' % n_args
        ]

class Literal:
    def __init__(self, lit):
        self.lit = lit

    def emit(self):
        return [
            'top++;',
            'stack[top] = (cell)%s;' % self.lit
        ]

class StringLiteral(Literal):
    def __init__(self, lit):
        super().__init__(lit)

class IntLiteral(Literal):
    def __init__(self, lit):
        super().__init__(lit)


class Custom:
    def __init__(self, code):
        self.code = code

    def emit(self):
        return self.code
