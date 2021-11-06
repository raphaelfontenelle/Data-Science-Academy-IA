# Agentes Lógicos

"""

Abrange Lógica Proposicional e de Primeira Ordem. Primeiro temos quatro
Tipos de dados importantes:

    KB            Uma classe abstrata que contém uma base de conhecimento de expressões lógicas
    KB_Agent      Classe abstrata que é subclasse de agentes.Agent
    Expr          Uma expressão lógica, importada de utils.py
    substitution  Implementado como um dicionário de pares key:value, {x: 1, y: x}

Atenção: algumas funções levam um Expr como argumento, e algumas tomam um KB.

Expressões lógicas podem ser criadas com Expr ou expr, importadas de utils, que adicionam 
a capacidade de escrever uma string que usa os conectores ==>, <==, <=> ou <= / =>. 
Mas tenha cuidado com a precedência. Consulte logic.ipynb para obter exemplos.

Em seguida, implementamos várias funções para fazer inferência lógica:

    pl_true          Avalia uma sentença lógica proposicional em um modelo
    tt_entails       Determina se uma declaração é vinculada a um KB
    pl_resolution    Faz resolução sobre frases proposicionais
    dpll_satisfiable Veja se uma sentença proposicional é satisfazível
    WalkSAT          Tenta encontrar uma solução para um conjunto de cláusulas

E algumas outras funções:

    to_cnf           Converte para forma conjuntiva normal
    unify            Faz a unificação de duas frases FOL
    diff, simp       Diferenciação simbólica e simplificação
"""

from utils import (
    removeall, unique, first, argmax, probability,
    isnumber, issequence, Symbol, Expr, expr, subexpressions
)
import agentes

import itertools
import random
from collections import defaultdict

# ______________________________________________________________________________


class KB:

    """Uma base de conhecimento para a qual você pode dizer e pedir frases."""

    def __init__(self, sentence=None):
        raise NotImplementedError

    def tell(self, sentence):
        "Adicione a frase ao KB."
        raise NotImplementedError

    def ask(self, query):
        """Retorna uma substituição que torna a consulta verdadeira ou, caso contrário, retorna False."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        "Produza todas as substituições que tornam a consulta verdadeira."
        raise NotImplementedError

    def retract(self, sentence):
        "Remover a sentença da KB."
        raise NotImplementedError


class PropKB(KB):

    "Um KB para lógica proposicional."

    def __init__(self, sentence=None):
        self.clauses = []
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        "Adicione as cláusulas da sentença ao KB."
        self.clauses.extend(conjuncts(to_cnf(sentence)))

    def ask_generator(self, query):
        "Valida a substituição vazia {} se KB envolve consulta; Senão não há resultados."
        if tt_entails(Expr('&', *self.clauses), query):
            yield {}

    def ask_if_true(self, query):
        "Retornar True se o KB envolve consulta, senão retorna False."
        for _ in self.ask_generator(query):
            return True
        return False

    def retract(self, sentence):
        "Remove as cláusulas da sentença da KB."
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)

# ______________________________________________________________________________


def KB_AgentProgram(KB):
    """Um programa genérico de agente baseado em conhecimento lógico."""
    steps = itertools.count()

    def program(percept):
        t = next(steps)
        KB.tell(make_percept_sentence(percept, t))
        action = KB.ask(make_action_query(t))
        KB.tell(make_action_sentence(action, t))
        return action

    def make_percept_sentence(self, percept, t):
        return Expr("Percept")(percept, t)

    def make_action_query(self, t):
        return expr("ShouldDo(action, {})".format(t))

    def make_action_sentence(self, action, t):
        return Expr("Did")(action[expr('action')], t)

    return program


def is_symbol(s):
    "Um string s é um símbolo se ele começa com um caracter alfabético."
    return isinstance(s, str) and s[:1].isalpha()


def is_var_symbol(s):
    "Um símbolo de variável lógica é uma string inicial-minúscula."
    return is_symbol(s) and s[0].islower()


def is_prop_symbol(s):
    """Um símbolo de lógica de proposição é uma string inicial maiúscula."""
    return is_symbol(s) and s[0].isupper()


def variables(s):

    """Retorna um conjunto das variáveis na expressão s."""
    return {x for x in subexpressions(s) if is_variable(x)}


def is_definite_clause(s):

    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return (is_symbol(consequent.op) and
                all(is_symbol(arg.op) for arg in conjuncts(antecedent)))
    else:
        return False


def parse_definite_clause(s):
    "Devolver os antecedentes eo consequente de uma cláusula definitiva."
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent

# Constante Exprs útil usado em exemplos e código:
A, B, C, D, E, F, G, P, Q, x, y, z = map(Expr, 'ABCDEFGPQxyz')


# ______________________________________________________________________________


def tt_entails(kb, alpha):
    """Tabelas-verdade para frases da KB proposicional
    Note que o 'kb' deve ser um Expr que é uma conjunção de cláusulas.
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    assert not variables(alpha)
    return tt_check_all(kb, alpha, prop_symbols(kb & alpha), {})


def tt_check_all(kb, alpha, symbols, model):
    "Auxiliary routine to implement tt_entails."
    if not symbols:
        if pl_true(kb, model):
            result = pl_true(alpha, model)
            assert result in (True, False)
            return result
        else:
            return True
    else:
        P, rest = symbols[0], symbols[1:]
        return (tt_check_all(kb, alpha, rest, extend(model, P, True)) and
                tt_check_all(kb, alpha, rest, extend(model, P, False)))


def prop_symbols(x):
    "Return a list of all propositional symbols in x."
    if not isinstance(x, Expr):
        return []
    elif is_prop_symbol(x.op):
        return [x]
    else:
        return list(set(symbol for arg in x.args for symbol in prop_symbols(arg)))


def tt_true(s):

    s = expr(s)
    return tt_entails(True, s)


def pl_true(exp, model={}):
    """Retorna True se a expressão lógica proposicional for verdadeira no modelo,
     e False se for falso. Se o modelo não especificar o valor para
     cada proposição, isto pode retornar nenhuma para indicar 'não óbvio';
     Isso pode acontecer mesmo quando a expressão é tautológica."""
    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor oo 'not equivalent'
        return pt != qt
    else:
        raise ValueError("Operador ilegal na expressão lógica" + str(exp))

# ______________________________________________________________________________

# Converter em Forma Normal Conjuntiva (CNF)


def to_cnf(s):
    """Converte uma sentença lógica proposicional em forma conjuntiva normal.
    Ou seja, para a forma: ((A | ~B | ...) & (B | C | ...) & ...) 
    >>> to_cnf('~(B | C)')
    (~B & ~C)
    """
    s = expr(s)
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implications(s)  
    s = move_not_inwards(s)  
    return distribute_and_over_or(s)  


def eliminate_implications(s):
    "Altera as implicações em forma equivalente com apenas &, |, e ~ como operadores lógicos."
    s = expr(s)
    if not s.args or is_symbol(s.op):
        return s 
    args = list(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2  
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    """Reescreva sentenças s movendo sinal de negação.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)"""
    s = expr(s)
    if s.op == '~':
        def NOT(b):
            return move_not_inwards(~b)
        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])  # ~~A ==> A
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *list(map(move_not_inwards, s.args)))


def distribute_and_over_or(s):
    """Dada uma sentença s consistindo de conjunções e disjunções
     de literais, devolver uma sentença equivalente em CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    s = expr(s)
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return False
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = first(arg for arg in s.args if arg.op == '&')
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


def associate(op, args):
    """Dada uma op associativa, retornar uma expressão com o mesmo
     significado como Expr (op, * args), ou seja, com instâncias aninhadas
     do mesmo grupo promovido ao nível superior.
    >>> associate('&', [(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)

_op_identity = {'&': True, '|': False, '+': 0, '*': 1}


def dissociate(op, args):
    """Dada uma op associativa, retornar um resultado da lista tal
     que Expr (op, * resultado) significa o mesmo que Expr (op, * args)."""
    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)
    collect(args)
    return result


def conjuncts(s):
    return dissociate('&', [s])


def disjuncts(s):
    return dissociate('|', [s])

# ______________________________________________________________________________


def pl_resolution(KB, alpha):
    "Resolução Propositional-lógica"
    clauses = KB.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    while True:
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i+1, n)]
        for (ci, cj) in pairs:
            resolvents = pl_resolve(ci, cj)
            if False in resolvents:
                return True
            new = new.union(set(resolvents))
        if new.issubset(set(clauses)):
            return False
        for c in new:
            if c not in clauses:
                clauses.append(c)


def pl_resolve(ci, cj):
    clauses = []
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                dnew = unique(removeall(di, disjuncts(ci)) +
                              removeall(dj, disjuncts(cj)))
                clauses.append(associate('|', dnew))
    return clauses

# ______________________________________________________________________________


class PropDefiniteKB(PropKB):

    "Um KB de cláusulas proposicionais definidas."

    def tell(self, sentence):
        "Adicione uma cláusula definitiva a esta KB."
        assert is_definite_clause(sentence), "Deve ser cláusula definitiva"
        self.clauses.append(sentence)

    def ask_generator(self, query):
        if pl_fc_entails(self.clauses, query):
            yield {}

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def clauses_with_premise(self, p):
        return [c for c in self.clauses
                if c.op == '==>' and p in conjuncts(c.args[0])]


def pl_fc_entails(KB, q):
    count = {c: len(conjuncts(c.args[0]))
             for c in KB.clauses
             if c.op == '==>'}
    inferred = defaultdict(bool)
    agenda = [s for s in KB.clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q:
            return True
        if not inferred[p]:
            inferred[p] = True
            for c in KB.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False


wumpus_world_inference = expr("(B11 <=> (P12 | P21))  &  ~B11")


horn_clauses_KB = PropDefiniteKB()
for s in "P==>Q; (L&M)==>P; (B&L)==>M; (A&P)==>L; (A&B)==>L; A;B".split(';'):
    horn_clauses_KB.tell(expr(s))

# ______________________________________________________________________________


def dpll_satisfiable(s):

    clauses = conjuncts(to_cnf(s))
    symbols = prop_symbols(s)
    return dpll(clauses, symbols, {})


def dpll(clauses, symbols, model):
    "See if the clauses are true in a partial model."
    unknown_clauses = []  
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, removeall(P, symbols), extend(model, P, value))
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, removeall(P, symbols), extend(model, P, value))
    if not symbols:
        raise TypeError("Argument should be of the type Expr.")
    P, symbols = symbols[0], symbols[1:]
    return (dpll(clauses, symbols, extend(model, P, True)) or
            dpll(clauses, symbols, extend(model, P, False)))


def find_pure_symbol(symbols, clauses):

    for s in symbols:
        found_pos, found_neg = False, False
        for c in clauses:
            if not found_pos and s in disjuncts(c):
                found_pos = True
            if not found_neg and ~s in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return s, found_pos
    return None, None


def find_unit_clause(clauses, model):

    for clause in clauses:
        P, value = unit_clause_assign(clause, model)
        if P:
            return P, value
    return None, None


def unit_clause_assign(clause, model):

    P, value = None, None
    for literal in disjuncts(clause):
        sym, positive = inspect_literal(literal)
        if sym in model:
            if model[sym] == positive:
                return None, None  
        elif P:
            return None, None      
        else:
            P, value = sym, positive
    return P, value


def inspect_literal(literal):
    if literal.op == '~':
        return literal.args[0], False
    else:
        return literal, True

# ______________________________________________________________________________


def WalkSAT(clauses, p=0.5, max_flips=10000):

    symbols = set(sym for clause in clauses for sym in prop_symbols(clause))
    model = {s: random.choice([True, False]) for s in symbols}
    for i in range(max_flips):
        satisfied, unsatisfied = [], []
        for clause in clauses:
            (satisfied if pl_true(clause, model) else unsatisfied).append(clause)
        if not unsatisfied:  
            return model
        clause = random.choice(unsatisfied)
        if probability(p):
            sym = random.choice(prop_symbols(clause))
        else:
            def sat_count(sym):
                model[sym] = not model[sym]
                count = len([clause for clause in clauses if pl_true(clause, model)])
                model[sym] = not model[sym]
                return count
            sym = argmax(prop_symbols(clause), key=sat_count)
        model[sym] = not model[sym]
    return None

# ______________________________________________________________________________


class HybridWumpusAgent(agentes.Agent):

    "An agent for the wumpus world that does logical inference. [Figure 7.20]"""

    def __init__(self):
        raise NotImplementedError


def plan_route(current, goals, allowed):
    raise NotImplementedError

# ______________________________________________________________________________


def SAT_plan(init, transition, goal, t_max, SAT_solver=dpll_satisfiable):

    def translate_to_SAT(init, transition, goal, time):
        clauses = []
        states = [state for state in transition]

        state_counter = itertools.count()
        for s in states:
            for t in range(time+1):
                state_sym[s, t] = Expr("State_{}".format(next(state_counter)))

        clauses.append(state_sym[init, 0])

        clauses.append(state_sym[goal, time])

        transition_counter = itertools.count()
        for s in states:
            for action in transition[s]:
                s_ = transition[s][action]
                for t in range(time):
                    action_sym[s, action, t] = Expr("Transition_{}".format(next(transition_counter)))

                    clauses.append(action_sym[s, action, t] |'==>'| state_sym[s, t])
                    clauses.append(action_sym[s, action, t] |'==>'| state_sym[s_, t + 1])

        for t in range(time+1):
            clauses.append(associate('|', [state_sym[s, t] for s in states]))

            for s in states:
                for s_ in states[states.index(s) + 1:]:
                    clauses.append((~state_sym[s, t]) | (~state_sym[s_, t]))

        for t in range(time):
            transitions_t = [tr for tr in action_sym if tr[2] == t]

            clauses.append(associate('|', [action_sym[tr] for tr in transitions_t]))

            for tr in transitions_t:
                for tr_ in transitions_t[transitions_t.index(tr) + 1 :]:
                    clauses.append(~action_sym[tr] | ~action_sym[tr_])

        return associate('&', clauses)

    def extract_solution(model):
        true_transitions = [t for t in action_sym if model[action_sym[t]]]
        true_transitions.sort(key=lambda x: x[2])
        return [action for s, action, time in true_transitions]

    for t in range(t_max):
        state_sym = {}
        action_sym = {}

        cnf = translate_to_SAT(init, transition, goal, t)
        model = SAT_solver(cnf)
        if model is not False:
            return extract_solution(model)
    return None


# ______________________________________________________________________________


def unify(x, y, s):
    if s is None:
        return None
    elif x == y:
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif isinstance(x, Expr) and isinstance(y, Expr):
        return unify(x.args, y.args, unify(x.op, y.op, s))
    elif isinstance(x, str) or isinstance(y, str):
        return None
    elif issequence(x) and issequence(y) and len(x) == len(y):
        if not x:
            return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s))
    else:
        return None


def is_variable(x):
    "A variable is an Expr with no args and a lowercase symbol as the op."
    return isinstance(x, Expr) and not x.args and x.op[0].islower()


def unify_var(var, x, s):
    if var in s:
        return unify(s[var], x, s)
    elif occur_check(var, x, s):
        return None
    else:
        return extend(s, var, x)


def occur_check(var, x, s):
    if var == x:
        return True
    elif is_variable(x) and x in s:
        return occur_check(var, s[x], s)
    elif isinstance(x, Expr):
        return (occur_check(var, x.op, s) or
                occur_check(var, x.args, s))
    elif isinstance(x, (list, tuple)):
        return first(e for e in x if occur_check(var, e, s))
    else:
        return False


def extend(s, var, val):
    "Copy the substitution s and extend it by setting var to val; return copy."
    s2 = s.copy()
    s2[var] = val
    return s2


def subst(s, x):
    if isinstance(x, list):
        return [subst(s, xi) for xi in x]
    elif isinstance(x, tuple):
        return tuple([subst(s, xi) for xi in x])
    elif not isinstance(x, Expr):
        return x
    elif is_var_symbol(x.op):
        return s.get(x, x)
    else:
        return Expr(x.op, *[subst(s, arg) for arg in x.args])


def fol_fc_ask(KB, alpha):
    raise NotImplementedError


def standardize_variables(sentence, dic=None):
    if dic is None:
        dic = {}
    if not isinstance(sentence, Expr):
        return sentence
    elif is_var_symbol(sentence.op):
        if sentence in dic:
            return dic[sentence]
        else:
            v = Expr('v_{}'.format(next(standardize_variables.counter)))
            dic[sentence] = v
            return v
    else:
        return Expr(sentence.op,
                    *[standardize_variables(a, dic) for a in sentence.args])

standardize_variables.counter = itertools.count()

# ______________________________________________________________________________


class FolKB(KB):

    def __init__(self, initial_clauses=[]):
        self.clauses = []  # inefficient: no indexing
        for clause in initial_clauses:
            self.tell(clause)

    def tell(self, sentence):
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception("Not a definite clause: {}".format(sentence))

    def ask_generator(self, query):
        return fol_bc_ask(self, query)

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def fetch_rules_for_goal(self, goal):
        return self.clauses


test_kb = FolKB(
    map(expr, ['Farmer(Mac)',
               'Rabbit(Pete)',
               'Mother(MrsMac, Mac)',
               'Mother(MrsRabbit, Pete)',
               '(Rabbit(r) & Farmer(f)) ==> Hates(f, r)',
               '(Mother(m, c)) ==> Loves(m, c)',
               '(Mother(m, r) & Rabbit(r)) ==> Rabbit(m)',
               '(Farmer(f)) ==> Human(f)',
               '(Mother(m, h) & Human(h)) ==> Human(m)'
               ]))

crime_kb = FolKB(
    map(expr,
             ['(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)',  # noqa
              'Owns(Nono, M1)',
              'Missile(M1)',
              '(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)',
              'Missile(x) ==> Weapon(x)',
              'Enemy(x, America) ==> Hostile(x)',
              'American(West)',
              'Enemy(Nono, America)'
              ]))


def fol_bc_ask(KB, query):
    return fol_bc_or(KB, query, {})


def fol_bc_or(KB, goal, theta):
    for rule in KB.fetch_rules_for_goal(goal):
        lhs, rhs = parse_definite_clause(standardize_variables(rule))
        for theta1 in fol_bc_and(KB, lhs, unify(rhs, goal, theta)):
            yield theta1


def fol_bc_and(KB, goals, theta):
    if theta is None:
        pass
    elif not goals:
        yield theta
    else:
        first, rest = goals[0], goals[1:]
        for theta1 in fol_bc_or(KB, subst(theta, first), theta):
            for theta2 in fol_bc_and(KB, rest, theta1):
                yield theta2


def diff(y, x):
    if y == x:
        return 1
    elif not y.args:
        return 0
    else:
        u, op, v = y.args[0], y.op, y.args[-1]
        if op == '+':
            return diff(u, x) + diff(v, x)
        elif op == '-' and len(y.args) == 1:
            return -diff(u, x)
        elif op == '-':
            return diff(u, x) - diff(v, x)
        elif op == '*':
            return u * diff(v, x) + v * diff(u, x)
        elif op == '/':
            return (v * diff(u, x) - u * diff(v, x)) / (v * v)
        elif op == '**' and isnumber(x.op):
            return (v * u ** (v - 1) * diff(u, x))
        elif op == '**':
            return (v * u ** (v - 1) * diff(u, x) +
                    u ** v * Expr('log')(u) * diff(v, x))
        elif op == 'log':
            return diff(u, x) / u
        else:
            raise ValueError("Unknown op: {} in diff({}, {})".format(op, y, x))


def simp(x):
    "Simplify the expression x."
    if isnumber(x) or not x.args:
        return x
    args = list(map(simp, x.args))
    u, op, v = args[0], x.op, args[-1]
    if op == '+':
        if v == 0:
            return u
        if u == 0:
            return v
        if u == v:
            return 2 * u
        if u == -v or v == -u:
            return 0
    elif op == '-' and len(args) == 1:
        if u.op == '-' and len(u.args) == 1:
            return u.args[0]  # --y ==> y
    elif op == '-':
        if v == 0:
            return u
        if u == 0:
            return -v
        if u == v:
            return 0
        if u == -v or v == -u:
            return 0
    elif op == '*':
        if u == 0 or v == 0:
            return 0
        if u == 1:
            return v
        if v == 1:
            return u
        if u == v:
            return u ** 2
    elif op == '/':
        if u == 0:
            return 0
        if v == 0:
            return Expr('Undefined')
        if u == v:
            return 1
        if u == -v or v == -u:
            return 0
    elif op == '**':
        if u == 0:
            return 0
        if v == 0:
            return 1
        if u == 1:
            return 1
        if v == 1:
            return u
    elif op == 'log':
        if u == 1:
            return 0
    else:
        raise ValueError("Unknown op: " + op)
    return Expr(op, *args)


def d(y, x):
    "Differentiate and then simplify."
    return simp(diff(y, x))
