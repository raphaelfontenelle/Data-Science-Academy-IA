# Módulo com diversas funções utilitárias

import bisect
import collections
import collections.abc
import functools
import operator
import os.path
import random
import math


# Funções de iterações

def sequence(iterable):
    "Coerce iterable to sequence, if it is not already one."
    return (iterable if isinstance(iterable, collections.abc.Sequence)
            else tuple(iterable))


def removeall(item, seq):
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]


def unique(seq):  
    return list(set(seq))


def count(seq):
    return sum(bool(x) for x in seq)


def product(numbers):
    result = 1
    for x in numbers:
        result *= x
    return result


def first(iterable, default=None):
    "Return the first element of an iterable or the next element of a generator; or default."
    try:
        return iterable[0]
    except IndexError:
        return default
    except TypeError:
        return next(iterable, default)


def is_in(elt, seq):
    return any(x is elt for x in seq)


# Argmin e Argmax

identity = lambda x: x

argmin = min
argmax = max


def argmin_random_tie(seq, key=identity):
    return argmin(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmax(shuffled(seq), key=key)


def shuffled(iterable):
    "Randomly shuffle a copy of iterable."
    items = list(iterable)
    random.shuffle(items)
    return items


# Funções estatísticas e matemáticas

def histogram(values, mode=0, bin_function=None):
    if bin_function:
        values = map(bin_function, values)

    bins = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1

    if mode:
        return sorted(list(bins.items()), key=lambda x: (x[1], x[0]),
                      reverse=True)
    else:
        return sorted(bins.items())


def dotproduct(X, Y):
    return sum(x * y for x, y in zip(X, Y))


def element_wise_product(X, Y):
    assert len(X) == len(Y)
    return [x * y for x, y in zip(X, Y)]


def matrix_multiplication(X_M, *Y_M):

    def _mat_mult(X_M, Y_M):

        assert len(X_M[0]) == len(Y_M)

        result = [[0 for i in range(len(Y_M[0]))] for j in range(len(X_M))]
        for i in range(len(X_M)):
            for j in range(len(Y_M[0])):
                for k in range(len(Y_M)):
                    result[i][j] += X_M[i][k] * Y_M[k][j]
        return result

    result = X_M
    for Y in Y_M:
        result = _mat_mult(result, Y)

    return result


def vector_to_diagonal(v):
    diag_matrix = [[0 for i in range(len(v))] for j in range(len(v))]
    for i in range(len(v)):
        diag_matrix[i][i] = v[i]

    return diag_matrix


def vector_add(a, b):
    return tuple(map(operator.add, a, b))



def scalar_vector_product(X, Y):
    return [X * y for y in Y]


def scalar_matrix_product(X, Y):
    return [scalar_vector_product(X, y) for y in Y]


def inverse_matrix(X):
    assert len(X) == 2
    assert len(X[0]) == 2
    det = X[0][0] * X[1][1] - X[0][1] * X[1][0]
    assert det != 0
    inv_mat = scalar_matrix_product(1.0/det, [[X[1][1], -X[0][1]], [-X[1][0], X[0][0]]])

    return inv_mat


def probability(p):
    "Return true with probability p."
    return p > random.uniform(0.0, 1.0)


def weighted_sample_with_replacement(seq, weights, n):
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]


def weighted_sampler(seq, weights):
    "Return a random-sample function that picks from seq weighted by weights."
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def rounder(numbers, d=4):
    "Round a single number, or sequence of numbers, to d decimal places."
    if isinstance(numbers, (int, float)):
        return round(numbers, d)
    else:
        constructor = type(numbers)     # Can be list, set, tuple, etc.
        return constructor(rounder(n, d) for n in numbers)


def num_or_str(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def normalize(dist):
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


def clip(x, lowest, highest):
    return max(lowest, min(x, highest))


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def step(x):
    return 1 if x >= 0 else 0

try:  
    from math import isclose
except ImportError:
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        "Return true if numbers a and b are close to each other."
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


#  Use functools.lru_cache memoization decorator

def memoize(fn, slot=None):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}

    return memoized_fn


def name(obj):
    "Try to find some reasonable name for the object."
    return (getattr(obj, 'name', 0) or getattr(obj, '__name__', 0) or
            getattr(getattr(obj, '__class__', 0), '__name__', 0) or
            str(obj))


def isnumber(x):
    "Is x a number?"
    return hasattr(x, '__int__')


def issequence(x):
    "Is x a sequence?"
    return isinstance(x, collections.abc.Sequence)


def print_table(table, header=None, sep='   ', numfmt='%g'):

    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))


def AIMAFile(components, mode='r'):
    "Open a file based at the AIMA root directory."
    aima_root = os.path.dirname(__file__)

    aima_file = os.path.join(aima_root, *components)

    return open(aima_file)


def DataFile(name, mode='r'):
    "Return a file in the AIMA /aima-data directory."
    return AIMAFile(['aima-data', name], mode)


# Expressões

# See https://docs.python.org/3/reference/expressions.html#operator-precedence
# See https://docs.python.org/3/reference/datamodel.html#special-method-names

class Expr(object):

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    # Operator overloads
    def __neg__(self):      return Expr('-', self)
    def __pos__(self):      return Expr('+', self)
    def __invert__(self):   return Expr('~', self)
    def __add__(self, rhs): return Expr('+', self, rhs)
    def __sub__(self, rhs): return Expr('-', self, rhs)
    def __mul__(self, rhs): return Expr('*', self, rhs)
    def __pow__(self, rhs): return Expr('**',self, rhs)
    def __mod__(self, rhs): return Expr('%', self, rhs)
    def __and__(self, rhs): return Expr('&', self, rhs)
    def __xor__(self, rhs): return Expr('^', self, rhs)
    def __rshift__(self, rhs):   return Expr('>>', self, rhs)
    def __lshift__(self, rhs):   return Expr('<<', self, rhs)
    def __truediv__(self, rhs):  return Expr('/',  self, rhs)
    def __floordiv__(self, rhs): return Expr('//', self, rhs)
    def __matmul__(self, rhs):   return Expr('@',  self, rhs)

    def __or__(self, rhs):
        "Allow both P | Q, and P |'==>'| Q."
        if isinstance(rhs, Expression):
            return Expr('|', self, rhs)
        else:
            return PartialExpr(rhs, self)

    def __radd__(self, lhs): return Expr('+',  lhs, self)
    def __rsub__(self, lhs): return Expr('-',  lhs, self)
    def __rmul__(self, lhs): return Expr('*',  lhs, self)
    def __rdiv__(self, lhs): return Expr('/',  lhs, self)
    def __rpow__(self, lhs): return Expr('**', lhs, self)
    def __rmod__(self, lhs): return Expr('%',  lhs, self)
    def __rand__(self, lhs): return Expr('&',  lhs, self)
    def __rxor__(self, lhs): return Expr('^',  lhs, self)
    def __ror__(self, lhs):  return Expr('|',  lhs, self)
    def __rrshift__(self, lhs):   return Expr('>>',  lhs, self)
    def __rlshift__(self, lhs):   return Expr('<<',  lhs, self)
    def __rtruediv__(self, lhs):  return Expr('/',  lhs, self)
    def __rfloordiv__(self, lhs): return Expr('//',  lhs, self)
    def __rmatmul__(self, lhs):   return Expr('@', lhs, self)

    def __call__(self, *args):
        "Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."
        if self.args:
            raise ValueError('can only do a call for a Symbol, not an Expr')
        else:
            return Expr(self.op, *args)

    def __eq__(self, other):
        "'x == y' evaluates to True or False; does not build an Expr."
        return (isinstance(other, Expr)
                and self.op == other.op
                and self.args == other.args)

    def __hash__(self): return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():       # f(x) or f(x, y)
            return '{}({})'.format(op, ', '.join(args)) if args else op
        elif len(args) == 1:        # -x or -(x + 1)
            return op + args[0]
        else:                       # (x - y)
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'


Number = (int, float, complex)
Expression = (Expr, Number)


def Symbol(name):
    "A Symbol is just an Expr with no args."
    return Expr(name)


def symbols(names):
    "Return a tuple of Symbols; names is a comma/whitespace delimited str."
    return tuple(Symbol(name) for name in names.replace(',', ' ').split())


def subexpressions(x):
    "Yield the subexpressions of an Expression (including x itself)."
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)


def arity(expression):
    "The number of sub-expressions in this expression."
    if isinstance(expression, Expr):
        return len(expression.args)
    else:  # expression is a number
        return 0



class PartialExpr:
    def __init__(self, op, lhs): self.op, self.lhs = op, lhs
    def __or__(self, rhs):       return Expr(self.op, self.lhs, rhs)
    def __repr__(self):          return "PartialExpr('{}', {})".format(self.op, self.lhs)


def expr(x):

    if isinstance(x, str):
        return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol))
    else:
        return x

infix_ops = '==> <== <=>'.split()


def expr_handle_infix_ops(x):

    for op in infix_ops:
        x = x.replace(op, '|' + repr(op) + '|')
    return x


class defaultkeydict(collections.defaultdict):

    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


# Filas: Stack, FIFOQueue, PriorityQueue

class Queue:

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


def Stack():
    return []


class FIFOQueue(Queue):

    def __init__(self):
        self.A = []
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]


class PriorityQueue(Queue):


    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)


# Distância

orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, -1)


def turn_left(heading):
    return turn_heading(heading, +1)


def distance(a, b):
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))


def distance2(a, b):
    "The square of the distance between two (x, y) points."
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def vector_clip(vector, lowest, highest):
    return type(vector)(map(clip, vector, lowest, highest))



class Bool(int):
    __str__ = __repr__ = lambda self: 'T' if self else 'F'

T = Bool(True)
F = Bool(False)
