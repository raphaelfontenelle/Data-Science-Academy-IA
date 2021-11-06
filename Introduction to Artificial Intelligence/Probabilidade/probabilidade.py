""" Modelos de Probabilidade """

from utils import (
    product, argmax, element_wise_product, matrix_multiplication,
    vector_to_diagonal, vector_add, scalar_vector_product, inverse_matrix,
    weighted_sample_with_replacement, isclose, probability, normalize
)
from logica import extend

import random
from collections import defaultdict
from functools import reduce

# ______________________________________________________________________________


def DTAgentProgram(belief_state):
    "Agente de decisão teórica"
    def program(percept):
        belief_state.observe(program.action, percept)
        program.action = argmax(belief_state.actions(), key = belief_state.expected_outcome_utility)
        return program.action
    program.action = None
    return program

# ______________________________________________________________________________


class ProbDist:

    """ Distribuição Discreta de Probabilidade.  Você nomeia a variável aleatória
     no construtor, e então atribui e consulta a probabilidade de valores."""

    def __init__(self, varname='?', freqs=None):
        """Se freqs é dado, é um dicionário de valor: pares de frequência,
         e o ProbDist então é normalizado."""
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        "Dado um valor, retornar P(valor)."
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        "Set P(val) = p."
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Verifique se as probabilidades de todos os valores somam 1.
         Retorna a distribuição normalizada.
         Aumenta um ZeroDivisionError se a soma dos valores for 0."""
        total = sum(self.prob.values())
        if not isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='%.3g'):
        """Mostre as probabilidades arredondadas e classificadas por
         De doctests portáteis."""
        return ', '.join([('%s: ' + numfmt) % (v, p)
                          for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P(%s)" % self.varname


class JointProbDist(ProbDist):
    """Uma probabilidade discreta distribui sobre um conjunto de variáveis."""

    def __init__(self, variables):
        self.prob = {}
        self.variables = variables
        self.vals = defaultdict(list)

    def __getitem__(self, values):
        "Dada uma tupla ou dict de valores, retornar P(valores)."
        values = event_values(values, self.variables)
        return ProbDist.__getitem__(self, values)

    def __setitem__(self, values, p):
        """Set P (valores) = p. Os valores podem ser uma tupla ou um dict; deve
         Têm um valor para cada uma das variáveis na articulação. Também acompanhar
         Dos valores que vimos até agora para cada variável."""
        values = event_values(values, self.variables)
        self.prob[values] = p
        for var, val in zip(self.variables, values):
            if val not in self.vals[var]:
                self.vals[var].append(val)

    def values(self, var):
        "Retorna o conjunto de valores possíveis para uma variável."
        return self.vals[var]

    def __repr__(self):
        return "P(%s)" % self.variables


def event_values(event, variables):
    """Retorna uma tupla dos valores das variáveis no evento."""
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])

# ______________________________________________________________________________


def enumerate_joint_ask(X, e, P):
    """Retornar uma distribuição de probabilidade sobre os valores da variável X,
     Dadas as observações {var: val} e, no método JointProbDist P. """
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)  # Distribuição de probabilidade para X, inicialmente vazia
    Y = [v for v in P.variables if v != X and v not in e]  # Variáveis ocultas.
    for xi in P.values(X):
        Q[xi] = enumerate_joint(Y, extend(e, X, xi), P)
    return Q.normalize()


def enumerate_joint(variables, e, P):
    """Retornar a soma dessas entradas em P consistente com e,
     As variáveis fornecidas são as variáveis restantes de P (aquelas não em e)."""
    if not variables:
        return P[e]
    Y, rest = variables[0], variables[1:]
    return sum([enumerate_joint(rest, extend(e, Y, y), P)
                for y in P.values(Y)])

# ______________________________________________________________________________


class BayesNet:

    "Rede bayesiana contendo apenas nós de variáveis booleanas."

    def __init__(self, node_specs=[]):
        "Os nós devem ser ordenados com os pais antes das crianças."
        self.nodes = []
        self.variables = []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """Adicione um nó à rede. Seus pais já devem estar no
         Sua variável não deve ser."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """Retornar o nó para a variável chamada var."""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)

    def variable_values(self, var):
        "Retorna o domínio de var."
        return [True, False]

    def __repr__(self):
        return 'BayesNet(%r)' % self.nodes


class BayesNode:

    """Uma distribuição de probabilidade condicional para uma variável booleana,
     P (X | pais). Parte de um BayesNet."""

    def __init__(self, X, parents, cpt):
        """X é um nome de variável, e os pais uma seqüência de variável
         Nomes ou uma seqüência separada por espaço. Cpt, o condicional
         Tabela de probabilidade, toma uma destas formas:

         * Um número, a probabilidade incondicional P (X = verdadeiro). Você pode
           Use este formulário quando não houver pais.

         * A dict {v: p, ...}, a distribuição de probabilidade condicional
           P (X = verdadeiro | pai = v) = p. Quando há apenas um pai.

         * A dict {(v1, v2, ...): p, ...}, a distribuição P (X = true |
           Parent1 = v1, parent2 = v2, ...) = p. Cada chave deve ter
           Valores como existem pais. Você pode usar este formulário sempre;
           Os dois primeiros são apenas conveniências.

         Em todos os casos, a probabilidade de X ser falso é deixada implícita,
         Pois segue de P (X = verdadeiro).
        """
        if isinstance(parents, str):
            parents = parents.split()

        # Armazenamos a tabela sempre na terceira forma acima.
        if isinstance(cpt, (float, int)):  
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """Retornar a probabilidade condicional
         P (X = valor | pais = valores_do_parente), onde os valores_de_parentes
         São os valores dos pais no evento. (O evento deve atribuir a cada
         Pai um valor.)"""
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """Amostra da distribuição para esta variável condicionada
         Em valores de eventos para as variáveis_de_mãe. Ou seja, retornar True / False
         Aleatoriamente de acordo com a probabilidade condicional dada a
         parentes."""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))


T, F = True, False

burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake',
     {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
])

# ______________________________________________________________________________


def enumeration_ask(X, e, bn):
    """Retornar a distribuição de probabilidade condicional da variável X
     Dadas evidências e, da BayesNet bn."""
    assert X not in e, "A variável de consulta deve ser distinta da evidência"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()


def enumerate_all(variables, e, bn):
    """Retorna a soma dessas entradas em P (variáveis | e {outros})
     Consistente com e, onde P é a distribuição conjunta representada
     Por bn, e e (outros) significa e restrito a outras variáveis de bn
     (Os que não sejam variáveis). Os pais devem preceder as crianças em variáveis."""
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))

# ______________________________________________________________________________


def elimination_ask(X, e, bn):
    """Calcule bn's P (X | e) por eliminação de variável."""
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize()


def is_hidden(var, X, e):
    "É var uma variável oculta ao consultar P (X | e)?"
    return var != X and var not in e


def make_factor(var, e, bn):
    """Retorne o fator para a distribuição conjunta var in bn dada e.
     Ou seja, a distribuição conjunta completa da bn, projetada de acordo com e,
     É o produto pontual desses fatores para as variáveis de bn."""
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {event_values(e1, variables): node.p(e1[var], e1)
           for e1 in all_events(variables, bn, e)}
    return Factor(variables, cpt)


def pointwise_product(factors, bn):
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)


def sum_out(var, factors, bn):
    "Eliminar var de todos os fatores somando sobre seus valores."
    result, var_factors = [], []
    for f in factors:
        (var_factors if var in f.variables else result).append(f)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result


class Factor:

    "Um fator em uma distribuição conjunta."

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn):
        "Multiplique dois fatores, combinando suas variáveis."
        variables = list(set(self.variables) | set(other.variables))
        cpt = {event_values(e, variables): self.p(e) * other.p(e)
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def sum_out(self, var, bn):
        "Faça um fator eliminando var por soma sobre seus valores."
        variables = [X for X in self.variables if X != var]
        cpt = {event_values(e, variables): sum(self.p(extend(e, var, val))
                                               for val in bn.variable_values(var))
               for e in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        "Retorne minhas probabilidades; Deve ser inferior a uma variável."
        assert len(self.variables) == 1
        return ProbDist(self.variables[0],
                        {k: v for ((k,), v) in self.cpt.items()})

    def p(self, e):
        "Procure meu valor tabulado para e."
        return self.cpt[event_values(e, self.variables)]


def all_events(variables, bn, e):
    "Rendimento cada maneira de estender e com valores para todas as variáveis."
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in all_events(rest, bn, e):
            for x in bn.variable_values(X):
                yield extend(e1, X, x)

# ______________________________________________________________________________


sprinkler = BayesNet([
    ('Cloudy', '', 0.5),
    ('Sprinkler', 'Cloudy', {T: 0.10, F: 0.50}),
    ('Rain', 'Cloudy', {T: 0.80, F: 0.20}),
    ('WetGrass', 'Sprinkler Rain',
     {(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00})])

# ______________________________________________________________________________


def prior_sample(bn):
    """Amostragem aleatória da distribuição da articulação completa da bn. O resultado
     É um valor {variable: value}. """
    event = {}
    for node in bn.nodes:
        event[node.variable] = node.sample(event)
    return event

# _________________________________________________________________________


def rejection_sampling(X, e, bn, N):
    """Estimar a distribuição de probabilidade da variável X dada
     Evidência e em BayesNet bn, usando N amostras. [Figura 14.14]
     Gera um ZeroDivisionError se todas as N amostras são rejeitadas,
     I.e., inconsistente com e."""
    counts = {x: 0 for x in bn.variable_values(X)}  
    for j in range(N):
        sample = prior_sample(bn)  
        if consistent_with(sample, e):
            counts[sample[X]] += 1
    return ProbDist(X, counts)


def consistent_with(event, evidence):
    "O evento é consistente com a evidência fornecida?"
    return all(evidence.get(k, v) == v
               for k, v in event.items())

# _________________________________________________________________________


def likelihood_weighting(X, e, bn, N):
    """Estimar a distribuição de probabilidade da variável X dada
     Evidência e em BayesNet bn. """
    W = {x: 0 for x in bn.variable_values(X)}
    for j in range(N):
        sample, weight = weighted_sample(bn, e)  
        W[sample[X]] += weight
    return ProbDist(X, W)


def weighted_sample(bn, e):
    """Exemplo de um evento de bn que é consistente com a evidência e;
     Retornar o evento e seu peso, a probabilidade de que o evento
     Concorda com as provas."""
    w = 1
    event = dict(e)  
    for node in bn.nodes:
        Xi = node.variable
        if Xi in e:
            w *= node.p(e[Xi], event)
        else:
            event[Xi] = node.sample(event)
    return event, w

# _________________________________________________________________________


def gibbs_ask(X, e, bn, N):
    assert X not in e, "A variável de consulta deve ser distinta da evidência"
    counts = {x: 0 for x in bn.variable_values(X)}  
    Z = [var for var in bn.variables if var not in e]
    state = dict(e) 
    for Zi in Z:
        state[Zi] = random.choice(bn.variable_values(Zi))
    for j in range(N):
        for Zi in Z:
            state[Zi] = markov_blanket_sample(Zi, state, bn)
            counts[state[X]] += 1
    return ProbDist(X, counts)


def markov_blanket_sample(X, e, bn):
    """Retornar uma amostra de P (X | mb) onde mb denota que a
     Variáveis no cobertor de Markov de X tomam seus valores do evento
     E (que deve atribuir um valor a cada). A manta de Markov de X é
     X pais, filhos e pais das crianças."""
    Xnode = bn.variable_node(X)
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        ei = extend(e, X, xi)
        Q[xi] = Xnode.p(xi, e) * product(Yj.p(ei[Yj.variable], ei)
                                         for Yj in Xnode.children)
    return probability(Q.normalize()[True])

# _________________________________________________________________________


class HiddenMarkovModel:

    """ Um modelo de markov oculto que leva modelo de transição e modelo de sensor como entradas"""

    def __init__(self, transition_model, sensor_model, prior=[0.5, 0.5]):
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.prior = prior

    def sensor_dist(self, ev):
        if ev is True:
            return self.sensor_model[0]
        else:
            return self.sensor_model[1]


def forward(HMM, fv, ev):
    prediction = vector_add(scalar_vector_product(fv[0], HMM.transition_model[0]),
                            scalar_vector_product(fv[1], HMM.transition_model[1]))
    sensor_dist = HMM.sensor_dist(ev)

    return normalize(element_wise_product(sensor_dist, prediction))


def backward(HMM, b, ev):
    sensor_dist = HMM.sensor_dist(ev)
    prediction = element_wise_product(sensor_dist, b)

    return normalize(vector_add(scalar_vector_product(prediction[0], HMM.transition_model[0]),
                                scalar_vector_product(prediction[1], HMM.transition_model[1])))


def forward_backward(HMM, ev, prior):
    """Algoritmo forward-backward para suavização. Calcula probabilidades posteriores
     De uma seqüência de estados dada uma seqüência de observações."""
    t = len(ev)
    ev.insert(0, None)  

    fv = [[0.0, 0.0] for i in range(len(ev))]
    b = [1.0, 1.0]
    bv = [b]    
    sv = [[0, 0] for i in range(len(ev))]

    fv[0] = prior

    for i in range(1, t + 1):
        fv[i] = forward(HMM, fv[i - 1], ev[i])
    for i in range(t, -1, -1):
        sv[i - 1] = normalize(element_wise_product(fv[i], b))
        b = backward(HMM, b, ev[i])
        bv.append(b)

    sv = sv[::-1]

    return sv

# _________________________________________________________________________


def fixed_lag_smoothing(e_t, HMM, d, ev, t):
    """Algoritmo de suavização com um intervalo de tempo fixo de passos 'd'.
     Algoritmo online que produz a nova estimativa suavizada se a observação
     Para novo passo de tempo é dado."""
    ev.insert(0, None)

    T_model = HMM.transition_model
    f = HMM.prior
    B = [[1, 0], [0, 1]]
    evidence = []

    evidence.append(e_t)
    O_t = vector_to_diagonal(HMM.sensor_dist(e_t))
    if t > d:
        f = forward(HMM, f, e_t)
        O_tmd = vector_to_diagonal(HMM.sensor_dist(ev[t - d]))
        B = matrix_multiplication(inverse_matrix(O_tmd), inverse_matrix(T_model), B, T_model, O_t)
    else:
        B = matrix_multiplication(B, T_model, O_t)
    t = t + 1

    if t > d:
        return [normalize(i) for i in matrix_multiplication([f], B)][0]
    else:
        return None

# _________________________________________________________________________


def particle_filtering(e, N, HMM):
    """Filtragem de partículas considerando duas variáveis de estados."""
    s = []
    dist = [0.5, 0.5]
    # Inicialização do estado
    s = ['A' if probability(dist[0]) else 'B' for i in range(N)]
    # Inicialização de peso
    w = [0 for i in range(N)]
    # PASSO 1 - Propagar um passo usando o modelo de transição dado estado anterior
    dist = vector_add(scalar_vector_product(dist[0], HMM.transition_model[0]),
                      scalar_vector_product(dist[1], HMM.transition_model[1]))
    # Atribuir o estado de acordo com a probabilidade
    s = ['A' if probability(dist[0]) else 'B' for i in range(N)]
    w_tot = 0
    # Calcular peso de importância dado evidência e
    for i in range(N):
        if s[i] == 'A':
            # P(U|A)*P(A)
            w_i = HMM.sensor_dist(e)[0] * dist[0]
        if s[i] == 'B':
            # P(U|B)*P(B)
            w_i = HMM.sensor_dist(e)[1] * dist[1]
        w[i] = w_i
        w_tot += w_i

    # Normalizar todos os pesos
    for i in range(N):
        w[i] = w[i] / w_tot

    # Limite pesos a 4 dígitos
    for i in range(N):
        w[i] = float("{0:.4f}".format(w[i]))

    # STEP 2
    s = weighted_sample_with_replacement(s, w, N)
    return s
