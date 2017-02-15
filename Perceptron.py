class Perceptron(object):
    def __init__(self, previous_layer=[]):
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        for n in self.previous_layer:
            n.next_layer.append(self)

        self.value = None

    def move_forward(self):
        raise NotImplemented



class Input(Perceptron):
    def __init__(self):
        Perceptron.__init__(self)

    def move_forward(self):
        if value is not None:
            self.value = value

class Add(Perceptron):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def move_forward(self):

def forward(output, sorted_nodes):
    for node in sorted_nodes:
        node.move_forward()

    return output.value()

def topological_sort(node_dict):
    
