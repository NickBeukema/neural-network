import math

def sigmoid(value):
    return 1 / (1 + (math.exp(-1 * value)))

def softmax(values):
    maxed_values = []

    sum = 0
    for i in values:
        sum += math.exp(i)

    for i in values:
        maxed_value = math.exp(i) / sum
        maxed_values.append(maxed_value)

    return maxed_values

def max(values):
    max_value = 0

    for value in values:
        if value > max_value:
            max_value = value

    max_array = [0] * len(values)

    for i in range(len(values)):
        if max_value == values[i]:
            max_array[i] = 1

    return max_array

def mean(values):
    total = 0

    for value in values:
        total += float(value)

    return float(total) / float(len(values))

def standard_deviation(values, m=None):
    if m is None:
        m = mean(values)

    new_set = []

    for value in values:
        new_set.append(math.pow((float(value) - m), 2))

    second_mean = mean(new_set)

    return math.sqrt(second_mean)

def calc_network_error(expected, results):
    error = 0

    for i in range(len(expected)):
        error += math.pow(expected[i] - results[i], 2) / 2

    return error
