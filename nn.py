import pdb
import random
import functions
import math
import parser
import grapher

class Node(object):

    def __init__(self, is_bias=False):
        self.up = []
        self.down = []
        self.is_bias = is_bias
        self.value = 0
        self.error = 0
        self.soft_max = 0

        if is_bias:
            self.value = 1

    def add_edge_up(self, edge):
        self.up.append(edge)

    def add_edge_down(self, edge):
        self.down.append(edge)

    def squash_value(self):
        self.value = functions.sigmoid(self.value)

    def calc_value(self):
        value = 0
        for edge in self.down:
            value += edge.down.value * edge.weight

        self.value = value


    # Calculate error for a hidden node
    def calc_hidden_error(self):

        # Derivative of the sigmoid function, getting the
        # original value
        value = self.value * (1 - self.value)

        weight_totals = 0

        for edge in self.up:
            weight_totals += edge.weight * edge.up.error

        self.error = value * weight_totals

    # Calculates the change in weight for all the edges
    # going downward from the current node and adds it to the
    # current weight change
    def calc_weight_down(self, learning_rate):
        for edge in self.down:
            edge.weight_change += learning_rate * self.error * edge.down.value


class Edge(object):

    # Weight variables to configure random range
    ## Random Offset
    RO = .5

    ## Random Multiplyer
    RM = .02

    def __init__(self, weight=0, down=None, up=None):
        self.weight = weight
        self.weight_change = 0
        self.up = up
        self.down = down

        # If defining an upward node, make sure to set
        # the current edge downward on the node
        if up is not None:
            up.add_edge_down(self)

        # If defining a downward node, make sure to set
        # the current edge upward on the down
        if down is not None:
            down.add_edge_up(self)

    def update_weight(self):
        self.weight = self.weight + self.weight_change
        self.weight_change = 0

class NodeLayer(object):

    def __init__(self, node_count=0, create_bias=True):
        self.nodes = []
        self.bias = None

        if node_count > 0:
            self.build_nodes(node_count, create_bias)

    # Gets values from current layer's nodes
    def get_values(self):
        values = []
        for node in self.nodes:
            values.append(node.value)

        return values

    def get_non_bias_nodes(self):
        nodes = []
        for node in self.nodes:
            if node.is_bias is False:
                nodes.append(node)

        return nodes

    def add_node(self, node):
        self.nodes.append(node)

        if node.is_bias:
            self.bias = node

    # Sets all nodes to the same error value, used
    # for output nodes
    def set_output_errors(self, value):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            node.error = node.soft_max * (1 - node.soft_max) * (value[i] - node.soft_max)

    # Triggers a calc_error() for each contained node,
    # used for hidden nodes
    def set_hidden_errors(self):
        for node in self.nodes:
            node.calc_hidden_error()

    # Sets values for each of the contained nodes
    def set_values(self, values):

        # Subtract one to avoid setting bias node
        for i in range(len(self.nodes) - 1):
            self.nodes[i].value = values[i]

    # Builds a given amount of nodes in the
    # current layer
    def build_nodes(self, node_count, add_bias=True):
        for i in range(node_count):
            self.add_node(Node())

        # Add bias node if needed
        if add_bias:
            bias_node = Node(is_bias=True)
            self.add_node(bias_node)

    # Connects every node in one layer to every
    # node in the next, expect the bias nodes
    # on the upper layer
    def connect_layer(self, upper_layer):
        for node in self.nodes:
            for upper_node in upper_layer.nodes:
                if upper_node.is_bias is False:
                    r = (random.random() - Edge.RO) * Edge.RM
                    # r = .5
                    edge = Edge(r,node,upper_node)

    # Take nodes contained by current layer and update weights_down
    # each of their edges by the define weight change on the edge
    def update_weights_down(self):
        for node in self.nodes:
            for edge in node.down:
                edge.update_weight()

    # Calculates the value for each contained node in current layer,
    # which assumes the values in the layer below are already set
    def calculate_values(self):
        for node in self.nodes:
            if node.is_bias is False:
                node.calc_value()

    # Takes every node in layer and uses the sigmoid function
    def squash_values(self):
        for node in self.nodes:
            if node.is_bias is False:
                node.squash_value()

    # Calculates the weight change for every nodes
    # edges going downward
    def calc_weights_down(self, learning_rate):
        for node in self.nodes:
            node.calc_weight_down(learning_rate)

    # Sets the softmax of the outputs to a variable on
    # each node
    def soft_max_outputs(self):
        max_values = functions.softmax(self.get_values())

        for i in range(len(self.nodes)):
            self.nodes[i].soft_max = max_values[i]

class Network(object):

    # Build a neural network based on a given amount
    # of inputs, hidden nodes, and outputs
    def __init__(self, input_count, hidden_layer_node_count, output_count, learning_rate=1):
        self.input_layer = NodeLayer(input_count)
        self.hidden_layer = NodeLayer(hidden_layer_node_count)
        self.output_layer = NodeLayer(output_count, create_bias=False)

        self.learning_rate = learning_rate
        self.error = 0

        self.input_layer.connect_layer(self.hidden_layer)
        self.hidden_layer.connect_layer(self.output_layer)

    # Triggers each layer to update their weights downward
    # since the input layer doesn't have downward edges,
    # there is no need to trigger them
    def update_weights(self):
        self.output_layer.update_weights_down()
        self.hidden_layer.update_weights_down()

    def feed_forward(self, values):
        self.input_layer.set_values(values)
        self.hidden_layer.calculate_values()
        self.hidden_layer.squash_values()
        self.output_layer.calculate_values()
        self.output_layer.soft_max_outputs()

    def get_outputs(self):
        return functions.softmax(self.output_layer.get_values())

    def get_inputs(self):
        return self.input_layer.get_values()

    def calc_total_error(self, expected):
        # Calculate error on entire network
        self.error = functions.calc_network_error(expected, self.get_outputs())

        # Calculate error on output nodes
        self.output_layer.set_output_errors(expected)

        # Calculate error on hidden nodes
        self.hidden_layer.set_hidden_errors()

    # Calculates the weight changes for all edges
    def calc_weight_changes(self):

        # First calculate the change for edges between
        # output and hidden nodes
        self.output_layer.calc_weights_down(self.learning_rate)

        # Then calculate the change for edges between
        # hidden and input nodes
        self.hidden_layer.calc_weights_down(self.learning_rate)

    def back_propagate(self, expected):
        self.calc_total_error(expected)
        self.calc_weight_changes()

    # Weight deltas are stored on the edges themselves,
    # this adds the deltas to the edge weights and clears
    # the deltas
    def update_weights(self):
        self.output_layer.update_weights_down()
        self.hidden_layer.update_weights_down()

    def categorize(self, data):
        self.feed_forward(data)
        results = self.get_outputs()
        maxed = functions.max(results)

        return maxed

    # Takes a set of data and tests the means squared error
    # of the output.
    def get_error_from_test(self, test_data):

        total_error = 0

        outputs_index = -1 * len(self.output_layer.nodes)

        for example in test_data:
            self.feed_forward(example[:outputs_index])
            total_error += functions.calc_network_error(example[outputs_index:], self.get_outputs())

        return total_error / len(test_data)

    # Categorizes a set of data and finds the average error
    # of the set of classifications
    def get_categorized_error_from_test(self, test_data):

        total_correct = 0
        outputs_index = -1 * len(self.output_layer.nodes)

        # Keep track of an array of incorrect [guess, actual]
        categorized_list = []

        for example in test_data:

            result = self.categorize(example[:outputs_index])



            if result == example[outputs_index:]:
                total_correct += 1
            else:
                categorized_list.append([result, example[outputs_index:]])

        error = 1 - (float(total_correct) / float(len(test_data)))

        return error


# A helper to run an online learning session, the network, data,
# the test set, and the number of epochs must be provided
def online_learn(network, data, test, epochs):

    training_errors = []
    testing_errors = []
    categorize_accuracy = []

    # Get the index of the first output value by checking
    # how many output nodes are in the network
    outputs_index = -1 * len(network.output_layer.nodes)

    # Begin Training
    for i in range(epochs):

        # Shuffle data before each epoch, because it's
        # online learning
        random.shuffle(data)

        error = 0

        # Feedforward and backpropagate
        for example in data:
            network.feed_forward(example[:outputs_index])
            network.back_propagate(example[outputs_index:])

            # Update weights between each example
            network.update_weights()

            # Keep track of total error of network to calculate
            # average error below
            error += network.error

        # Calculate different error stats
        training_errors.append(error / len(data))
        testing_errors.append(network.get_error_from_test(test))
        categorize_accuracy.append(network.get_categorized_error_from_test(test))

    return {
        "train": training_errors,
        "test": testing_errors,
        "categorize_accuracy": categorize_accuracy
    }

# A helper to run a batch learning session, the network, data,
# the test set, and the number of epochs must be provided
def batch_learn(network, data, test, epochs):

    training_errors = []
    testing_errors = []
    categorize_accuracy = []

    # Get the index of the first output value by checking
    # how many output nodes are in the network
    outputs_index = -1 * len(network.output_layer.nodes)

    # Begin Training
    for i in range(epochs):

        error = 0

        # Feedforward and backpropagate
        for example in data:
            network.feed_forward(example[:outputs_index])
            network.back_propagate(example[outputs_index:])

            # Keep track of total error of network to calculate
            # average error below
            error += network.error

        # After running through all examples, update the weights
        network.update_weights()

        # Calculate different error stats
        training_errors.append(error / len(data))
        testing_errors.append(network.get_error_from_test(test))
        categorize_accuracy.append(network.get_categorized_error_from_test(test))

    return {
        "train": training_errors,
        "test": testing_errors,
        "categorize_accuracy": categorize_accuracy
    }

# Preset data formatted for the game data
def game_data():
    translations = parser.game_data_format()
    all_data = parser.parse('../game-data.csv', translations)

    return {
        'train': all_data[:-5],
        'test': all_data[-5:],
        'values': {
            'inputs': 6,
            'hidden': 4,
            'outputs': 4
        },
        'batch_learning_rate': 1,
        'online_learning_rate': 1
    }

# Preset data formatted for the iris data
def iris_data():
    translations = parser.iris_data_format()
    all_data = parser.parse('data/iris_data.csv', translations)

    random.shuffle(all_data)

    # Used to cut the data in two, one for training,
    # the other for testing
    length = len(all_data) // 3

    return {
        # 'train': all_data[:length],
        'train': all_data[:105],
        # 'test': all_data[length:],
        'test': all_data[105:],
        'values': {
            'inputs': 4,
            'hidden': 3,
            'outputs': 3
        },
        'batch_learning_rate': .5,
        'online_learning_rate': .1
    }


import time
start_time = time.time()

# Get data
data = iris_data()
values = data['values']


# Run batch learning
# network = Network(values['inputs'], values['hidden'], values['outputs'], learning_rate=data['batch_learning_rate'])
# batch = batch_learn(network, data['train'], data['test'], 150)

# Graph error statistics
# grapher.graph_line(batch, 'Batch Learning Statistics')
#
#
# # Run online learning
network2 = Network(values['inputs'], values['hidden'], values['outputs'], learning_rate=data['online_learning_rate'])
online = online_learn(network2, data['train'], data['test'], 200)

print("--- %s seconds ---" % (time.time() - start_time))
#
# # Graph error statistics
# grapher.graph_line(online, 'Online Learning Statistics')
