import math
import random
import copy
import json

in_layer = [0]*2
hidden_layer = [0]*6
out_layer = [0]*1

bias = [0.35, 0.60]

learning_rate = 0.5


def link(a, b):
    return [[random.random() for r in range(len(a))] for i in range(len(b))]


#in_hidden_w = [[0.15, 0.2], [0.25, 0.3]]
#hidden_out_w = [[0.4, 0.45], [0.5, 0.55]]

in_hidden_w = link(in_layer, hidden_layer)
hidden_out_w = link(hidden_layer, out_layer)


def test():
    return "Hallo Welt!"


def load_model():
    pass


def save_model(file):
    with open(file, "w+") as f:
        js = {
            'first_layer_weights': in_hidden_w,
            'second_layer_weights': hidden_out_w
        }
        f.write(json.dumps(js))


def fire(in_values):

    # feed forward for 2 first layers
    for hidden in range(len(hidden_layer)):

        weights = in_hidden_w[hidden]
        output = bias[0]

        for w in range(len(weights)):
            output = output + weights[w] * in_values[w]

        hidden_layer[hidden] = 1 / (1+math.e**(-output))
        # print("hidden: {}; value: {}".format(hidden, output))
        # print("after squashing: {}".format(hidden_layer[hidden]))

    for out in range(len(out_layer)):

        weights = hidden_out_w[out]
        output = bias[1]

        for w in range(len(weights)):
            output = output + weights[w] * hidden_layer[w]

        out_layer[out] = 1 / (1+math.e**(-output))

    return out_layer


def test_xor(x, y):
    result = fire([x, y])[0]

    if 0.5 - abs(result-0.5) < 0.01:
        print("absolute certain (nn is > 99% sure)")

    elif 0.5 - abs(result-0.5) < 0.1:
        print("very certain (nn is > 90% sure)")

    elif 0.5 - abs(result-0.5) < 0.3:
        print("rather certain (nn is > 70% sure)")

    elif 0.5 - abs(result-0.5) < 0.5:
        print("not certain (nn is > 50% sure)")

    print("{} ^ {} = {} ({})".format(x, y, 1 if result > 0.5 else 0, result))


def train(ins, out):
    in_layer = ins
    guess = fire(ins)
    error = 0.0

    for i in range(len(out)):
        error += (out_layer[i]-out[i])**2

    total_error = error/2

    derived_total_errors_to_outs = []
    derived_out_to_nets = []

    changes_hidden = copy.deepcopy(hidden_out_w)

    for i in range(len(guess)):
        derived_total_errors_to_outs.append(-(out[i]-out_layer[i]))
        derived_out_to_nets.append(out_layer[i]*(1-out_layer[i]))

        for j in range(len(hidden_layer)):
            net_to_w = hidden_layer[j]
            change = derived_total_errors_to_outs[i] * \
                derived_out_to_nets[i]*net_to_w
            changes_hidden[i][j] = hidden_out_w[i][j] - change*0.5

    derived_total_erros_to_outs_hidden = []

    for i in range(len(hidden_layer)):

        err = 0.0

        for j in range(len(guess)):

            # print("hidden: " + str(i), "out: " + str(j))
            # print(derived_total_errors_to_outs[j])
            # print(derived_out_to_nets[j])
            # print(hidden_out_w[j][i])
            # print()

            e = derived_total_errors_to_outs[j] * \
                derived_out_to_nets[j] * hidden_out_w[j][i]

            err += e

        derived_total_erros_to_outs_hidden.append(err)
        #print("hidden: " + str(i) + " err: " + str(err))

    # print(derived_total_erros_to_outs_hidden)

    for i in range(len(hidden_layer)):
        for j in range(len(in_layer)):
            error_w_j = derived_total_erros_to_outs_hidden[i] * (
                hidden_layer[i] * (1-hidden_layer[i])) * in_layer[j]

            in_hidden_w[i][j] -= error_w_j*0.5

    for i in range(len(hidden_out_w)):
        for j in range(len(hidden_out_w[i])):
            hidden_out_w[i][j] = changes_hidden[i][j]


def test_fire(count, threshold):

    succ = 0.0

    for i in range(count):
        ins = [random.randrange(0, 2), random.randrange(0, 2)]
        out = True if fire(ins)[0] > threshold else False

        if out is (ins[0] is not ins[1]):
            succ += 1

    print("Success rate for {} tries: {:.2f} %".format(
        count,  (succ / count)*100))


def train_nn(count):
    print("training XOR {} times".format(count))
    for i in range(0, count):
        train([0, 1], [1])
        train([1, 0], [1])
        train([0, 0], [0])
        train([1, 1], [0])

# test_fire(1000)
# print("Yes" if fire([0, 1])[0] > 0.5 else "No")

# test_xor(x, y)
# train_nn(count)
# test_fire(count, threshold)


save_model('model.json')
