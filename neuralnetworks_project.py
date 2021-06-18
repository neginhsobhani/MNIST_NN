import time
import numpy as np
import matplotlib.pyplot as plt
import math


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
def get_data():
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))

    # Reading The Test Set
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))
    return test_set, train_set


# Plotting an image
# show_image(train_set[1][0])
# plt.show()

# def sigmoid(x):
#     return np.divide(1.0, np.add(1.0, np.exp(-x)))


def sigmoid(z):
    try:
        res = np.divide(1.0, np.add(1.0,  np.exp(-z)))
    except OverflowError:
        res = 0
    return res


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# SECOND STEP - FEEDFORWARD
def feedforward(ff_input):
    z1 = (weights[0] @ ff_input[0]) + biases[0]
    a1 = np.asarray([sigmoid(z[0]) for z in z1]).reshape((16, 1))
    z2 = (weights[1] @ a1) + biases[1]
    a2 = np.asarray([sigmoid(z[0]) for z in z2]).reshape((16, 1))
    z3 = (weights[2] @ a2) + biases[2]
    a3 = np.asarray([sigmoid(z[0]) for z in z3]).reshape((10, 1))
    return [a1, a2, a3], [z1, z2, z3]


# STEP3 - BACK PROPAGATION
def back_propagation(grad_w, grad_b, grad_a, a, z, img, w):
    # for j in range(10):
    #     for k in range(16):
    #         grad_w[2][j, k] += a[1][k, 0] * sigmoid_derivative(z[2][j, 0]) * (2 * a[2][j, 0] - 2 * img[1][j, 0])
    grad_w[2] += (2 * sigmoid_derivative(z[2]) * (a[2] - img[1])) @ np.transpose(a[1])

    # for j in range(10):
    #     grad_b[2][j, 0] += sigmoid_derivative(z[2][j, 0]) * (2 * a[2][j, 0] - 2 * img[1][j, 0])
    grad_b[2] += 2 * sigmoid_derivative(z[2]) * (a[2] - img[1])

    # for j in range(16):
    #     for k in range(10):
    #         grad_a[1][j, 0] += weights[2][k, j] * sigmoid_derivative(z[2][k, 0]) * (2 * a[2][k, 0] - 2 * img[1][k])
    grad_a[1] += np.transpose(w[2]) @ (2 * sigmoid_derivative(z[2]) * (a[2] - img[1]))

    # for j in range(16):
    #     for k in range(16):
    #         grad_w[1][j, k] += grad_a[1][j, 0] * sigmoid_derivative(z[1][j, 0]) * a[0][k, 0]
    grad_w[1] += grad_a[1] * sigmoid_derivative(z[1]) @ np.transpose(a[0])

    # for j in range(16):
    #     grad_b[1][j, 0] += sigmoid_derivative(z[1][j, 0]) * grad_a[1][j, 0]
    grad_b[1] += grad_a[1] * sigmoid_derivative(z[1])

    # for j in range(16):
    #     for k in range(10):
    #         grad_a[0][j, 0] += weights[1][k, j] * sigmoid_derivative(z[1][k, 0]) * grad_a[1][k, 0]
    grad_a[0] += np.transpose(w[1]) @ (grad_a[1] * sigmoid_derivative(z[1]))

    # for j in range(16):
    #     for k in range(784):
    #         grad_w[0][j, k] += grad_a[0][j, 0] * sigmoid_derivative(z[0][j, 0]) * img[0][k]
    grad_w[0] += grad_a[0] * sigmoid_derivative(z[0]) @ np.transpose(img[0])

    # for j in range(16):
    #     grad_b[0][j, 0] += sigmoid_derivative(z[0][j, 0]) * grad_a[0][j, 0]
    grad_b[0] += grad_a[0] * sigmoid_derivative(z[0])

    return grad_w, grad_b, grad_a


def calculate_accuracy(sets, sample_num):
    true_guesses = 0
    for image in range(sample_num):
        guess = np.argmax(feedforward(sets[image])[0][-1])
        label = np.argmax(sets[image][1])
        # true_guesses = true_guesses + 1 if guess == label else true_guesses
        if guess == label:
            true_guesses += 1
    accuracy = true_guesses / sample_num
    return accuracy


def train_network(epoch_num, train_set, sample_num, batch_size):
    errors = []
    accuracy = []
    for i in range(epoch_num):
        epoch_cost = 0
        np.random.shuffle(train_set)

        # batch_count = 0
        for j in range(0, sample_num, batch_size):
            # batch_count += 1
            batch = train_set[j:j + batch_size]

            grad_w3 = np.zeros((10, 16))
            grad_w2 = np.zeros((16, 16))
            grad_w1 = np.zeros((16, 784))
            grad_w = [grad_w1, grad_w2, grad_w3]

            grad_b1 = np.zeros((16, 1))
            grad_b2 = np.zeros((16, 1))
            grad_b3 = np.zeros((10, 1))
            grad_b = [grad_b1, grad_b2, grad_b3]

            grad_a1 = np.zeros((16, 1))
            grad_a2 = np.zeros((16, 1))
            grad_a3 = np.zeros((10, 1))
            grad_a = [grad_a1, grad_a2, grad_a3]

            for img in range(batch_size):
                a, z = feedforward(batch[img])
                grad_w, grad_b, grad_a = back_propagation(grad_w, grad_b, grad_a, a, z, batch[img], weights)
                c = 0
                for x in range(10):
                    c += (batch[img][1][x, 0] - a[2][x, 0]) ** 2
                epoch_cost += c

            for x in range(3):
                weights[x] -= (learning_rate*(grad_w[x] / batch_size))
                biases[x] -= (learning_rate*(grad_b[x] / batch_size))

        errors.append(epoch_cost / sample_num)
        accuracy.append(calculate_accuracy(train_set, sample_num))
        print("*EPOCH completed*")
        # plotting the average cost
    plt.plot(errors, 'b')
    plt.xlabel("Epoch", color='blue')
    plt.ylabel("Average Cost", color='blue')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    w0 = np.random.randn(16, 784)
    w1 = np.random.randn(16, 16)
    w2 = np.random.randn(10, 16)
    weights = [w0, w1, w2]

    b0 = np.zeros((16, 1))
    b1 = np.zeros((16, 1))
    b2 = np.zeros((10, 1))
    biases = [b0, b1, b2]
    learning_rate = 1
    epoch_num = 5
    batch_size = 5
    print("Reading from the dataset")
    test_set, train_set = get_data()
    print("Calculating accuracy of Feedforward")
    accuracy = calculate_accuracy(train_set, len(train_set))
    print("Initial Accuracy: ", accuracy * 100, "%")

    print("Training the network")
    start = time.time()
    train_network(epoch_num, train_set, len(train_set), batch_size)
    stop = time.time()
    print('Training process completed in {} seconds'.format(round(stop - start)))
    accuracy_train = calculate_accuracy(train_set, len(train_set))
    print("Accuracy of trained network on train test: ", accuracy_train * 100, "%")
    accuracy_test = calculate_accuracy(test_set, len(test_set))
    print("Accuracy of trained network on test set: ", accuracy_test * 100, "%")
