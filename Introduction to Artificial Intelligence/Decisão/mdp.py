""" Markov Decision Processes e Iteração de Valor

Primeiro, definimos um MDP e o caso especial de um GridMDP, no qual
estados são dispostos em uma grade bidimensional. Também representamos uma política
como um dicionário de pares {state: action} e uma função Utility como
dicionário de pares {state: number}. Em seguida, definimos os algoritmos de iteração de
valor e política."""

from utils import argmax, vector_add, print_table
from grid import orientations, turn_right, turn_left

import random


class MDP:

    """Um Processo de Decisão de Markov, definido por um estado inicial, modelo de transição,
     e função de recompensa. """

    def __init__(self, init, actlist, terminals, gamma=.9):
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = {}

    def R(self, state):
        "Retorna uma recompensa numérica para este estado."
        return self.reward[state]

    def T(self, state, action):
        """Modelo de transição. De um estado e uma ação, retorna uma lista
         de pares (probabilidade, resultado-estado)."""
        raise NotImplementedError

    def actions(self, state):
        """Conjunto de ações que podem ser executadas neste estado. Por padrão, uma
         lista fixa de ações, exceto para estados terminais."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist


class GridMDP(MDP):

    """Uma grade bidimensional MDP. Tudo que você tem a fazer é
     especificar a grade como uma lista de listas de recompensas; Use Nenhum para um obstáculo
     (Estado inacessível). Além disso, você deve especificar os estados do terminal.
     Uma acção é um vector unitário (x, y); por exemplo. (1, 0) significa mover-se para leste."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()  
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Retorna o estado que resulta de ir nessa direção."
        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Converte um mapeamento de (x, y) para v em uma grade [[..., v, ...]]."""
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {
            (1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

# ______________________________________________________________________________

""" 
Um ambiente de grade 4x3 que apresenta o agente com um problema de decisão seqüencial.
"""

sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None,  -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])

# ______________________________________________________________________________


def value_iteration(mdp, epsilon=0.001):
    "Resolvendo um MDP por iteração de valor."
    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    """Dado um MDP e uma função de utilidade U, determinar a melhor política,
     como um mapeamento do estado para a ação."""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    "A utilidade esperada de fazer um em estado s, de acordo com o MDP e U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])

# ______________________________________________________________________________


def policy_iteration(mdp):
    "Resolve um MDP por iteração de políticas"
    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi


def policy_evaluation(pi, U, mdp, k=20):
    """Retorna um utilitário atualizado mapeamento U de cada estado no MDP para seu
     Utilitário, usando uma aproximação (iteração de política modificada)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
    return U

__doc__ += """
>>> pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))

>>> sequential_decision_environment.to_arrows(pi)
[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '>', '^', '<']]

>>> print_table(sequential_decision_environment.to_arrows(pi))
>   >      >   .
^   None   ^   .
^   >      ^   <

>>> print_table(sequential_decision_environment.to_arrows(policy_iteration(sequential_decision_environment)))
>   >      >   .
^   None   ^   .
^   >      ^   <
"""
