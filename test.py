import numpy as np
from PIL import Image
import os
from random import shuffle
import pickle

def compute_loss(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def load_and_process_image(image_path, size=(64, 64), augment=False):
    img = Image.open(image_path).convert('RGB').resize(size)
    if augment:
        angle = np.random.uniform(-30, 30)
        img = img.rotate(angle)
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        brightness = np.random.uniform(0.7, 1.3)
        img = Image.eval(img, lambda x: x * brightness)
        contrast = np.random.uniform(0.8, 1.2)
        img = Image.eval(img, lambda x: x * contrast)
    img_array = np.array(img).flatten() / 255.0
    return img_array

def load_dataset(cat_dir, dog_dir, size=(64, 64), augment=False):
    X, y = [], []
    for img_name in os.listdir(cat_dir):
        if img_name.endswith(('.jpg', '.png')):
            img_path = os.path.join(cat_dir, img_name)
            X.append(load_and_process_image(img_path, size, augment))
            y.append(0)
            if augment:
                for _ in range(4):
                    X.append(load_and_process_image(img_path, size, True))
                    y.append(0)
    for img_name in os.listdir(dog_dir):
        if img_name.endswith(('.jpg', '.png')):
            img_path = os.path.join(dog_dir, img_name)
            X.append(load_and_process_image(img_path, size, augment))
            y.append(1)
            if augment:
                for _ in range(4):
                    X.append(load_and_process_image(img_path, size, True))
                    y.append(1)
    indices = np.arange(len(X))
    shuffle(indices)
    return np.array(X)[indices], np.array(y)[indices]

def initialize_weights(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size1))
    W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2.0 / hidden_size2)
    b3 = np.zeros((1, hidden_size3))
    W4 = np.random.randn(hidden_size3, hidden_size4) * np.sqrt(2.0 / hidden_size3)
    b4 = np.zeros((1, hidden_size4))
    W5 = np.random.randn(hidden_size4, output_size) * np.sqrt(2.0 / hidden_size4)
    b5 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

def leaky_relu(x, alpha=0.02):
    return np.where(x > 0, x, x * alpha)


def leaky_relu_derivative(x, alpha=0.02):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):
    Z1 = np.dot(X, W1) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = leaky_relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = leaky_relu(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = leaky_relu(Z4)
    Z5 = np.dot(A4, W5) + b5
    A5 = sigmoid(Z5)
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5

def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, learning_rate, dropout_rate=0.5, adam_params=None, t=1):
    m = X.shape[0]
    y = y.reshape(-1, 1)

    if adam_params is None:
        adam_params = {
            'mW1': 0, 'vW1': 0, 'mb1': 0, 'vb1': 0,
            'mW2': 0, 'vW2': 0, 'mb2': 0, 'vb2': 0,
            'mW3': 0, 'vW3': 0, 'mb3': 0, 'vb3': 0,
            'mW4': 0, 'vW4': 0, 'mb4': 0, 'vb4': 0,
            'mW5': 0, 'vW5': 0, 'mb5': 0, 'vb5': 0
        }

    dropout_mask1 = np.random.binomial(1, 1 - dropout_rate, size=A1.shape) / (1 - dropout_rate)
    dropout_mask2 = np.random.binomial(1, 1 - dropout_rate, size=A2.shape) / (1 - dropout_rate)
    dropout_mask3 = np.random.binomial(1, 1 - dropout_rate, size=A3.shape) / (1 - dropout_rate)
    dropout_mask4 = np.random.binomial(1, 1 - dropout_rate, size=A4.shape) / (1 - dropout_rate)
    A1 *= dropout_mask1
    A2 *= dropout_mask2
    A3 *= dropout_mask3
    A4 *= dropout_mask4

    dZ5 = A5 - y
    dW5 = np.dot(A4.T, dZ5) / m
    db5 = np.sum(dZ5, axis=0, keepdims=True) / m

    dA4 = np.dot(dZ5, W5.T)
    dZ4 = dA4 * leaky_relu_derivative(A4) * dropout_mask4
    dW4 = np.dot(A3.T, dZ4) / m
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * leaky_relu_derivative(A3) * dropout_mask3
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * leaky_relu_derivative(A2) * dropout_mask2
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * leaky_relu_derivative(A1) * dropout_mask1
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    lambda_reg = 0.002

    for param, dparam, mkey, vkey in [
        (W1, dW1, 'mW1', 'vW1'), (b1, db1, 'mb1', 'vb1'),
        (W2, dW2, 'mW2', 'vW2'), (b2, db2, 'mb2', 'vb2'),
        (W3, dW3, 'mW3', 'vW3'), (b3, db3, 'mb3', 'vb3'),
        (W4, dW4, 'mW4', 'vW4'), (b4, db4, 'mb4', 'vb4'),
        (W5, dW5, 'mW5', 'vW5'), (b5, db5, 'mb5', 'vb5')
    ]:
        adam_params[mkey] = beta1 * adam_params[mkey] + (1 - beta1) * dparam
        adam_params[vkey] = beta2 * adam_params[vkey] + (1 - beta2) * (dparam ** 2)
        m_hat = adam_params[mkey] / (1 - beta1 ** t)
        v_hat = adam_params[vkey] / (1 - beta2 ** t)
        param -= learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + lambda_reg * param if 'W' in mkey else 0)

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params

def save_model(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params, filepath='model.pkl'):
    model = {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
        'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4,
        'W5': W5, 'b5': b5, 'adam_params': adam_params
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath='model.pkl'):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return (model['W1'], model['b1'], model['W2'], model['b2'],
            model['W3'], model['b3'], model['W4'], model['b4'],
            model['W5'], model['b5'], model['adam_params'])

def train(X, y, hidden_size1=1024, hidden_size2=512, hidden_size3=256, hidden_size4=128, epochs=5000, learning_rate=0.001, batch_size=16, augment=True):
    input_size = X.shape[1]
    output_size = 1
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = initialize_weights(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size)
    adam_params = None

    train_split = int(0.8 * X.shape[0])
    val_split = int(0.9 * X.shape[0])
    indices = np.arange(X.shape[0])
    shuffle(indices)
    X_train, X_val, X_test = X[indices[:train_split]], X[indices[train_split:val_split]], X[indices[val_split:]]
    y_train, y_val, y_test = y[indices[:train_split]], y[indices[train_split:val_split]], y[indices[val_split:]]

    best_loss = float('inf')
    patience = 200
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5 = forward_propagation(X_batch, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
            W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params = backward_propagation(
                X_batch, y_batch, Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5,
                W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, learning_rate, adam_params=adam_params, t=epoch+1)

        _, _, _, _, _, _, _, _, _, A5_val = forward_propagation(X_val, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
        loss = compute_loss(y_val, A5_val)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Validation Loss: {loss:.4f}')

        if loss < best_loss:
            best_loss = loss
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy(), W4.copy(), b4.copy(), W5.copy(), b5.copy())
            patience_counter = 0
            save_model(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params, 'best_model.pkl')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = best_weights
                break

        if epoch % 1000 == 0 and epoch > 0:
            learning_rate *= 0.5
            print(f"Learning rate reduced to: {learning_rate}")

    _, _, _, _, _, _, _, _, _, A5_test = forward_propagation(X_test, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
    test_loss = compute_loss(y_test, A5_test)
    print(f"Test Loss: {test_loss:.4f}")

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params

def predict(image_path, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, size=(64, 64)):
    img = load_and_process_image(image_path, size)
    img = img.reshape(1, -1)
    _, _, _, _, _, _, _, _, _, A5 = forward_propagation(img, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
    return "Dog" if A5[0, 0] > 0.5 else "Cat"

cat_dir = "/cat"
dog_dir = "/dog"
test_image = "/test/test_image.jpg"
test_image2 = "/test/test_image2.jpg"

X, y = load_dataset(cat_dir, dog_dir, augment=True)

W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params = train(X, y, hidden_size1=1024, hidden_size2=512, hidden_size3=256, hidden_size4=128, epochs=5000, learning_rate=0.001, batch_size=16)

save_model(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params, 'final_model.pkl')

W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, adam_params = load_model('best_model.pkl')

result = predict(test_image, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
print(f"Prediction: {result}")

if os.path.exists(test_image2):
    result2 = predict(test_image2, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
    print(f"Prediction for second image: {result2}")
