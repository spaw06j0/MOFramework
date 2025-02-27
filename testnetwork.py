import pynet
import numpy as np

epoch = 10
batch_size = 256
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_labels.npy')
train_data = train_data / 255.0 - 0.5
train_label = train_label.astype(np.int32)
# one hot encoding
train_label = np.eye(10)[train_label]
train_data = pynet.Matrix(train_data)
train_label = pynet.Matrix(train_label)

test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_labels.npy')
test_data = test_data / 255.0 - 0.5
test_label = test_label.astype(np.int32)
# one hot encoding
test_label = np.eye(10)[test_label]
test_data = pynet.Matrix(test_data)
test_label = pynet.Matrix(test_label)

# train_data, train_label = pynet.load_mnist_data('./data/train_data.npy', './data/train_labels.npy', 60000)
# test_data, test_label = pynet.load_mnist_data('./data/test_data.npy', './data/test_labels.npy', 10000)
# print(np.array(train_data).shape)
# print(np.array(train_label).shape)

network = pynet.Network([
    pynet.Linear(784, 128, True, True),
    pynet.Sigmoid(),
    pynet.Linear(128, 10, True, True),
])

optimizer = pynet.SGD(0.003, 0.9)
loss_fn = pynet.CategoricalCrossentropy()

num_batches = train_data.getRow() // batch_size
# print(num_batches)

for e in range(epoch):
    print(f"Epoch {e + 1} started")
    total_loss = 0.0
    for b in range(num_batches):
        # when b = batch_size -> [b * batch_size :]
        batch_data = train_data.slice(b * batch_size, (b + 1) * batch_size)
        batch_label = train_label.slice(b * batch_size, (b + 1) * batch_size)
        predictions = network(batch_data)
        loss = loss_fn(predictions, batch_label)
        loss_gradient = loss_fn.backward()
        layer_gradients = network.backward(loss_gradient)
        optimizer.apply_gradient(network, layer_gradients)
        total_loss += loss.mean()
        if b % 100 == 0:
            print(f"Epoch {e + 1}/{epoch}, Batch {b}/{num_batches}, Loss: {loss.mean()}")
    print(f"Epoch {e + 1} finished, Average Loss: {total_loss / num_batches}")
    test_predictions = network(test_data)
    accuracy = pynet.compute_accuracy(test_predictions, test_label)
    print(f"Epoch {e + 1} finished, Test accuracy: {accuracy * 100}%")









