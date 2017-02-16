import numpy as np

### abstract class
class Node:
    def __init__(self, previous_layer=[]):
        self.previous_layer = previous_layer
        self.next_layer = []

        self.value = None
        self.gradients = {}

        #set this node to output of previous nodes
        for n in self.previous_layer:
            n.next_layer.append(self)

    def move_forward(self):
        raise NotImplementedError

    def move_backward(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def move_forward(self):
        pass

    def move_backward(self):
        self.gradients = { self: 0 }

        for n in self.next_layer:
            self.gradients[self] += n.gradients[self]

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def move_forward(self):
        inputs = self.previous_layer[0].value
        weights = self.previous_layer[1].value
        bias = self.previous_layer[2].value

        self.value = (np.dot(inputs, weights) + bias)

    def move_backward(self):
        self.gradients = {
            n : np.zeros_like(n.value) for n in self.previous_layer
        }

        for n in self.next_layer:
            grad_cost = n.gradients[self]
            self.gradients[self.previous_layer[0]] += np.dot(grad_cost,
                self.previous_layer[1].value.T)
            self.gradients[self.previous_layer[1]] += \
                np.dot(self.previous_layer[0].value.T, grad_cost)
            self.gradients[self.previous_layer[2]] += np.sum(grad_cost,
                axis = 0, keepdims = False)

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def move_forward(self):
        input_value = self.previous_layer[0].value
        self.value = self.sigmoid(input_value)

    def move_backward(self):
        self.gradients = {
            n: np.zeros_like(n.value) for n in self.previous_layer
        }

        for n in self.next_layer:
            grad_cost = n.gradients[self]
            d_sigmoid = self.value * (1 - self.value)
            self.gradients[self.previous_layer[0]] += (d_sigmoid * grad_cost)

class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def move_forward(self):
        y = self.previous_layer[0].value.reshape(-1, 1)
        a = self.previous_layer[1].value.reshape(-1, 1)

        ### number of elements to mean
        self.m = self.previous_layer[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff ** 2)

    def move_backward(self):
        self.gradients[self.previous_layer[0]] = (2 / self.m) * self.diff
        self.gradients[self.previous_layer[1]] = (-2 / self.m) * self.diff

### implements forward and backward pass
def round_trip(graph):
    for node in graph:
        node.move_forward()

    for node in graph[::-1]:
        node.move_backward()

def topological_sort(node_dict):
    ### Node dict is structured as node : node_value
    ### Kahn's algorithm from Wikipedia

    input_nodes = [n for n in node_dict.keys()]

    L = []
    S = set(input_nodes)
    nodes = list(input_nodes)
    G = {}

    ### Create graph
    while len(nodes) > 0:
        n = nodes.pop(0)

        if n not in G:
            G[n] = { 'in': set(), 'out': set() }
        for m in n.next_layer:
            if m not in G:
                G[m] = { 'in': set(), 'out': set() }
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    ### While loop from Kahn
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = node_dict[n]

        L.append(n)

        for m in n.next_layer:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)

            if len(G[m]['in']) == 0:
                S.add(m)

        ### assume this graph has no cycles

    return L

def stochastic_gradient_descent_update(trainables, learning_rate = .01):
    for t in trainables:
        partial = t.gradients[t]
        t.value -= learning_rate * partial

'''
def forward(output, sorted_nodes):
    for node in sorted_nodes:
        node.move_forward()

    return output.value()

def grad_descent_update(x, delta, learning_rate):
    x -= (delta * learning_rate)
    return x

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def move_forward(self):
        self.value = sum([x.value for x in self.previous_layer])

class Mult(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def move_forward(self):
        summa = 1

        for n in self.previous_layer:
            summa *= n.value

        self.value = summa
'''
