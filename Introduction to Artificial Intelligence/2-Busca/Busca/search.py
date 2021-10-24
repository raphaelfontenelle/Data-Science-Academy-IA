# Esse módulo contém a implementação de diversos algoritmos de busca que estudamos anteriormente

from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability,
    weighted_sample_with_replacement, memoize, print_table, DataFile, Stack,
    FIFOQueue, PriorityQueue, name, distance
)
from grid import *

from collections import defaultdict
import math
import random
import sys
import bisect

infinity = float('inf')


class Problem(object):

    """A classe abstrata para um problema formal. Você deve implementar subclasse
     para os métodos ações e resultados, e possivelmente
     __init__, goal_test e path_cost. Em seguida, você cria instâncias
     da sua subclasse e então resolve os problemas com as várias funções de busca."""

    def __init__(self, initial, goal=None):
        """O construtor especifica o estado inicial e possivelmente um esdado objetivo, 
         se houver uma meta única. O construtor da sua subclasse pode adicionar
         outros argumentos."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Retornar as ações que podem ser executadas no dado
         estado. O resultado normalmente seria uma lista, mas se houver
         muitas ações, considere levá-las de uma em uma
         iteração, ao invés de construí-los todos de uma vez."""
        raise NotImplementedError

    def result(self, state, action):
        """Retorna o estado que resulta da execução do dado
         Ação em um determinado estado. A acção deve ser uma das
         Auto-ações (estado)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Retornar verdadeiro se o estado for uma meta. O método padrão compara o
         estado para self.goal ou verifica o estado em self.goal se é uma
         lista, conforme especificado no construtor."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Retorna o custo de um caminho de solução que chega ao estado2 a partir
         do estado1 via ação, assumindo o custo c para chegar até estado1. Se o problema
         é tal que o caminho não importa, esta função só vai olhar para
         estado2. Se o caminho não importa, ele considerará c e talvez estado1
         e ação. O método padrão custa 1 para cada etapa do caminho."""
        return c + 1

    def value(self, state):
        raise NotImplementedError
# ______________________________________________________________________________


class Node:

    """Um nó em uma árvore de pesquisa. Contém um ponteiro para o pai (o nó
     que este é um sucessor) e para o estado real para este nó. Note
     que se um estado é atingido por dois caminhos, então há dois nós com
     o mesmo estado. Inclui também a ação que nos levou a este estado, e
     O path_cost total (também conhecido como g) para alcançar o nó. Outras funções
     podem adicionar um valor f e h; Ver best_first_graph_search e astar_search para
     uma explicação de como os valores f e h são tratados. Você não precisará
     de subclasse esta classe."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "[Figure 3.10]"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________


class SimpleProblemSolvingAgentProgram:

    """Agente de Resolução de Problemas"""

    def __init__(self, initial_state=None):
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError


# Uninformed Search algorithms


def tree_search(problem, frontier):
    frontier.append(Node(problem.initial))
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def graph_search(problem, frontier):
    frontier.append(Node(problem.initial))
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)
    return None


def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first."
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first."
    return tree_search(problem, Stack())


def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, Stack())


def breadth_first_search(problem):
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = FIFOQueue()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def uniform_cost_search(problem):
    return best_first_graph_search(problem, lambda node: node.path_cost)


def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


# Informed (Heuristic) Search

greedy_best_first_graph_search = best_first_graph_search


def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


# Other search algorithms


def recursive_best_first_search(problem, h=None):
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0   
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result


def hill_climbing(problem):
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors,
                                     key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state


def exp_schedule(k=20, lam=0.005, limit=100):
    "One possible schedule function for simulated annealing"
    return lambda t: (k * math.exp(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current
        neighbors = current.expand(problem)
        if not neighbors:
            return current
        next = random.choice(neighbors)
        delta_e = problem.value(next.state) - problem.value(current.state)
        if delta_e > 0 or probability(math.exp(delta_e / T)):
            current = next


def and_or_graph_search(problem):

    def or_search(state, problem, path):
        if problem.goal_test(state):
            return []
        if state in path:
            return None
        for action in problem.actions(state):
            plan = and_search(problem.result(state, action),
                              problem, path + [state, ])
            if plan is not None:
                return [action, plan]

    def and_search(states, problem, path):
        "returns plan in form of dictionary where we take action plan[s] if we reach state s"  # noqa
        plan = {}
        for s in states:
            plan[s] = or_search(s, problem, path)
            if plan[s] is None:
                return None
        return plan

    return or_search(problem.initial, problem, [])


class OnlineDFSAgent:

    def __init__(self, problem):
        self.problem = problem
        self.s = None
        self.a = None
        self.untried = defaultdict(list)
        self.unbacktracked = defaultdict(list)
        self.result = {}

    def __call__(self, percept):
        s1 = self.update_state(percept)
        if self.problem.goal_test(s1):
            self.a = None
        else:
            if s1 not in self.untried.keys():
                self.untried[s1] = self.problem.actions(s1)
            if self.s is not None:
                if s1 != self.result[(self.s, self.a)]:
                    self.result[(self.s, self.a)] = s1
                    unbacktracked[s1].insert(0, self.s)
            if len(self.untried[s1]) == 0:
                if len(self.unbacktracked[s1]) == 0:
                    self.a = None
                else:
                    unbacktracked_pop = self.unbacktracked[s1].pop(0)  # noqa
                    for (s, b) in self.result.keys():
                        if self.result[(s, b)] == unbacktracked_pop:
                            self.a = b
                            break
            else:
                self.a = self.untried[s1].pop(0)
        self.s = s1
        return self.a

    def update_state(self, percept):
        return percept

# ______________________________________________________________________________


class OnlineSearchProblem(Problem):

    def __init__(self, initial, goal, graph):
        self.initial = initial
        self.goal = goal
        self.graph = graph

    def actions(self, state):
        return self.graph.dict[state].keys()

    def output(self, state, action):
        return self.graph.dict[state][action]

    def h(self, state):
        return self.graph.least_costs[state]

    def c(self, s, a, s1):
        return 1

    def update_state(self, percept):
        raise NotImplementedError

    def goal_test(self, state):
        if state == self.goal:
            return True
        return False


class LRTAStarAgent:

    def __init__(self, problem):
        self.problem = problem
        self.H = {}
        self.s = None
        self.a = None

    def __call__(self, s1):    
        if self.problem.goal_test(s1):
            self.a = None
            return self.a
        else:
            if s1 not in self.H:
                self.H[s1] = self.problem.h(s1)
            if self.s is not None:
                self.H[self.s] = min(self.LRTA_cost(self.s, b, self.problem.output(self.s, b), self.H)
                                     for b in self.problem.actions(self.s))

            costs = [self.LRTA_cost(s1, b, self.problem.output(s1, b), self.H)
                     for b in self.problem.actions(s1)]
            self.a = list(self.problem.actions(s1))[costs.index(min(costs))]

            self.s = s1
            return self.a

    def LRTA_cost(self, s, a, s1, H):
        print(s, a, s1)
        if s1 is None:
            return self.problem.h(s)
        else:
            try:
                return self.problem.c(s, a, s1) + self.H[s1]
            except:
                return self.problem.c(s, a, s1) + self.problem.h(s1)


# Algoritmo Genético


def genetic_search(problem, fitness_fn, ngen=1000, pmut=0.1, n=20):

    s = problem.initial_state
    states = [problem.result(s, a) for a in problem.actions(s)]
    random.shuffle(states)
    return genetic_algorithm(states[:n], problem.value, ngen, pmut)


def genetic_algorithm(population, fitness_fn, ngen=1000, pmut=0.1):
    "[Figure 4.8]"
    for i in range(ngen):
        new_population = []
        for i in len(population):
            fitnesses = map(fitness_fn, population)
            p1, p2 = weighted_sample_with_replacement(population, fitnesses, 2)
            child = p1.mate(p2)
            if random.uniform(0, 1) < pmut:
                child.mutate()
            new_population.append(child)
        population = new_population
    return argmax(population, key=fitness_fn)


class GAState:

    "Abstract class for individuals in a genetic search."

    def __init__(self, genes):
        self.genes = genes

    def mate(self, other):
        "Return a new individual crossing self and other."
        c = random.randrange(len(self.genes))
        return self.__class__(self.genes[:c] + other.genes[c:])

    def mutate(self):
        "Change a few of my genes."
        raise NotImplementedError


# Graphs and Graph Problems


class Graph:

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        "Make a digraph into an undirected graph by adding symmetric edges."
        for a in list(self.dict.keys()):
            for (b, distance) in self.dict[a].items():
                self.connect1(b, a, distance)

    def connect(self, A, B, distance=1):
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        "Add a link from A to B of given distance, in one direction only."
        self.dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):

        links = self.dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        "Return a list of nodes in the graph."
        return list(self.dict.keys())


def UndirectedGraph(dict=None):
    "Build a Graph where every edge (including future ones) goes both ways."
    return Graph(dict=dict, directed=False)


def RandomGraph(nodes=list(range(10)), min_links=2, width=400, height=300,
                curvature=lambda: random.uniform(1.1, 1.5)):

    g = UndirectedGraph()
    g.locations = {}
    for node in nodes:
        g.locations[node] = (random.randrange(width), random.randrange(height))
    for i in range(min_links):
        for node in nodes:
            if len(g.get(node)) < min_links:
                here = g.locations[node]

                def distance_to_node(n):
                    if n is node or g.get(node, n):
                        return infinity
                    return distance(g.locations[n], here)
                neighbor = argmin(nodes, key=distance_to_node)
                d = distance(g.locations[neighbor], here) * curvature()
                g.connect(node, neighbor, int(d))
    return g


romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))
romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))


vacumm_world = Graph(dict(
    State_1 = dict(Suck = ['State_7', 'State_5'], Right = ['State_2']),
    State_2 = dict(Suck = ['State_8', 'State_4'], Left = ['State_2']),
    State_3 = dict(Suck = ['State_7'], Right = ['State_4']),
    State_4 = dict(Suck = ['State_4', 'State_2'], Left = ['State_3']),
    State_5 = dict(Suck = ['State_5', 'State_1'], Right = ['State_6']),
    State_6 = dict(Suck = ['State_8'], Left = ['State_5']),
    State_7 = dict(Suck = ['State_7', 'State_3'], Right = ['State_8']),
    State_8 = dict(Suck = ['State_8', 'State_6'], Left = ['State_7'])
    ))

one_dim_state_space = Graph(dict(
    State_1=dict(Right='State_2'),
    State_2=dict(Right='State_3', Left='State_1'),
    State_3=dict(Right='State_4', Left='State_2'),
    State_4=dict(Right='State_5', Left='State_3'),
    State_5=dict(Right='State_6', Left='State_4'),
    State_6=dict(Left='State_5')
    ))
one_dim_state_space.least_costs = dict(
    State_1=8,
    State_2=9,
    State_3=2,
    State_4=2,
    State_5=4,
    State_6=3)


australia_map = UndirectedGraph(dict(
    T=dict(),
    SA=dict(WA=1, NT=1, Q=1, NSW=1, V=1),
    NT=dict(WA=1, Q=1),
    NSW=dict(Q=1, V=1)))
australia_map.locations = dict(WA=(120, 24), NT=(135, 20), SA=(135, 30),
                               Q=(145, 20), NSW=(145, 32), T=(145, 42),
                               V=(145, 37))


class GraphProblem(Problem):

    "The problem of searching a graph from one node to another."

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        "The actions at a graph node are just its neighbors."
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        "The result of going to a neighbor is just that neighbor."
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or infinity)

    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity


class GraphProblemStochastic(GraphProblem):
 

    def result(self, state, action):
        return self.graph.get(state, action)

    def path_cost():
        raise NotImplementedError


# ______________________________________________________________________________


class NQueensProblem(Problem):


    def __init__(self, N):
        self.N = N
        self.initial = [None] * N

    def actions(self, state):
        "In the leftmost empty column, try all non-conflicting rows."
        if state[-1] is not None:
            return []  
        else:
            col = state.index(None)
            return [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def result(self, state, row):
        "Place the next queen at the given row."
        col = state.index(None)
        new = state[:]
        new[col] = row
        return new

    def conflicted(self, state, row, col):
        "Would placing a queen at (row, col) conflict with anything?"
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        "Would putting two queens in (row1, col1) and (row2, col2) conflict?"
        return (row1 == row2 or  
                col1 == col2 or  
                row1 - col1 == row2 - col2 or 
                row1 + col1 == row2 + col2)  

    def goal_test(self, state):
        "Check if all columns filled, no conflicts."
        if state[-1] is None:
            return False
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))



# Inverse Boggle: Search for a high-scoring Boggle board. 


ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

cubes16 = ['FORIXB', 'MOQABJ', 'GURILW', 'SETUPL',
           'CMPDAE', 'ACITAO', 'SLCRAE', 'ROMASH',
           'NODESW', 'HEFIYE', 'ONUDTK', 'TEVIGN',
           'ANEDVZ', 'PINESH', 'ABILYT', 'GKYLEU']


def random_boggle(n=4):
    """Return a random Boggle board of size n x n.
    We represent a board as a linear list of letters."""
    cubes = [cubes16[i % 16] for i in range(n * n)]
    random.shuffle(cubes)
    return list(map(random.choice, cubes))


boyan_best = list('RSTCSDEIAEGNLRPEATESMSSID')


def print_boggle(board):
    "Print the board in a 2-d array."
    n2 = len(board)
    n = exact_sqrt(n2)
    for i in range(n2):

        if i % n == 0 and i > 0:
            print()
        if board[i] == 'Q':
            print('Qu', end=' ')
        else:
            print(str(board[i]) + ' ', end=' ')
    print()


def boggle_neighbors(n2, cache={}):
    if cache.get(n2):
        return cache.get(n2)
    n = exact_sqrt(n2)
    neighbors = [None] * n2
    for i in range(n2):
        neighbors[i] = []
        on_top = i < n
        on_bottom = i >= n2 - n
        on_left = i % n == 0
        on_right = (i+1) % n == 0
        if not on_top:
            neighbors[i].append(i - n)
            if not on_left:
                neighbors[i].append(i - n - 1)
            if not on_right:
                neighbors[i].append(i - n + 1)
        if not on_bottom:
            neighbors[i].append(i + n)
            if not on_left:
                neighbors[i].append(i + n - 1)
            if not on_right:
                neighbors[i].append(i + n + 1)
        if not on_left:
            neighbors[i].append(i - 1)
        if not on_right:
            neighbors[i].append(i + 1)
    cache[n2] = neighbors
    return neighbors


def exact_sqrt(n2):
    "If n2 is a perfect square, return its square root, else raise error."
    n = int(math.sqrt(n2))
    assert n * n == n2
    return n

# _____________________________________________________________________________


class Wordlist:

    def __init__(self, file, min_len=3):
        lines = file.read().upper().split()
        self.words = [word for word in lines if len(word) >= min_len]
        self.words.sort()
        self.bounds = {}
        for c in ALPHABET:
            c2 = chr(ord(c) + 1)
            self.bounds[c] = (bisect.bisect(self.words, c),
                              bisect.bisect(self.words, c2))

    def lookup(self, prefix, lo=0, hi=None):
        words = self.words
        if hi is None:
            hi = len(words)
        i = bisect.bisect_left(words, prefix, lo, hi)
        if i < len(words) and words[i].startswith(prefix):
            return i, (words[i] == prefix)
        else:
            return None, False

    def __contains__(self, word):
        return self.lookup(word)[1]

    def __len__(self):
        return len(self.words)

# _____________________________________________________________________________


class BoggleFinder:

    wordlist = None  

    def __init__(self, board=None):
        if BoggleFinder.wordlist is None:
            BoggleFinder.wordlist = Wordlist(DataFile("EN-text/wordlist.txt"))
        self.found = {}
        if board:
            self.set_board(board)

    def set_board(self, board=None):
        "Set the board, and find all the words in it."
        if board is None:
            board = random_boggle()
        self.board = board
        self.neighbors = boggle_neighbors(len(board))
        self.found = {}
        for i in range(len(board)):
            lo, hi = self.wordlist.bounds[board[i]]
            self.find(lo, hi, i, [], '')
        return self

    def find(self, lo, hi, i, visited, prefix):
        if i in visited:
            return
        wordpos, is_word = self.wordlist.lookup(prefix, lo, hi)
        if wordpos is not None:
            if is_word:
                self.found[prefix] = True
            visited.append(i)
            c = self.board[i]
            if c == 'Q':
                c = 'QU'
            prefix += c
            for j in self.neighbors[i]:
                self.find(wordpos, hi, j, visited, prefix)
            visited.pop()

    def words(self):
        "The words found."
        return list(self.found.keys())

    scores = [0, 0, 0, 0, 1, 2, 3, 5] + [11] * 100

    def score(self):
        "The total score for the words found, according to the rules."
        return sum([self.scores[len(w)] for w in self.words()])

    def __len__(self):
        "The number of words found."
        return len(self.found)

# _____________________________________________________________________________


def boggle_hill_climbing(board=None, ntimes=100, verbose=True):
    finder = BoggleFinder()
    if board is None:
        board = random_boggle()
    best = len(finder.set_board(board))
    for _ in range(ntimes):
        i, oldc = mutate_boggle(board)
        new = len(finder.set_board(board))
        if new > best:
            best = new
            if verbose:
                print(best, _, board)
        else:
            board[i] = oldc  
    if verbose:
        print_boggle(board)
    return board, best


def mutate_boggle(board):
    i = random.randrange(len(board))
    oldc = board[i]
    board[i] = random.choice(random.choice(cubes16))
    return i, oldc



# Código para comparar vários algoritmos de busca


class InstrumentedProblem(Problem):


    def __init__(self, problem):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def value(self, state):
        return self.problem.value(state)

    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def __repr__(self):
        return '<%4d/%4d/%4d/%s>' % (self.succs, self.goal_tests,
                                     self.states, str(self.found)[:4])


def compare_searchers(problems, header,
                      searchers=[breadth_first_tree_search,
                                 breadth_first_search,
                                 depth_first_graph_search,
                                 iterative_deepening_search,
                                 depth_limited_search,
                                 recursive_best_first_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        searcher(p)
        return p
    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)


def compare_graph_searchers():
    """Imprime uma tabela de resultados de pesquisa."""
    compare_searchers(problems=[GraphProblem('Arad', 'Bucharest', romania_map),
                                GraphProblem('Oradea', 'Neamt', romania_map),
                                GraphProblem('Q', 'WA', australia_map)],
                      header=['Searcher', 'romania_map(Arad, Bucharest)',
                              'romania_map(Oradea, Neamt)', 'australia_map'])
