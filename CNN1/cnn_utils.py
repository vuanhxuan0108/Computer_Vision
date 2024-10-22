import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r") # Đọc vào tập dữ liệu huấn luyện dạng file .h5
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # Các đặc trưng của tập dữ liệu huấn luyện
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # Cột nhãn của tập dữ liệu huấn luyện

    test_dataset = h5py.File('datasets/test_signs.h5', "r") # Đọc vào tập dữ liệu kiểm tra dạng file .h5
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # Các đặc trưng của tập dữ liệu kiểm tra
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # Cột nhãn của tập dữ liệu kiểm tra

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # Chuyển đổi kích thước tập nhãn của dữ liệu huấn luyện từ (m hàng, n cột) thành (1 hàng, m cột)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # Chuyển đổi kích thước tập nhãn của dữ liệu kiểm tra từ (m hàng, n cột) thành (1 hàng, m cột)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[0] # Số mẫu trong tập huấn luyện
    mini_batches = []
    np.random.seed(seed)

    # Bước 1: Trộn ngẫu nhiên (X, Y)
    permutation = list(np.random.permutation(m)) # Tạo ra danh sách hoán vị ngẫu nhiên của các số trong đoạn [1, m], ví dụ: [1, 5, 3, 7, 113, ...., 123, 8, 24, m]
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Bước 2: Chia (shuffled_X, shuffled_Y) thành các mini_batch có kích thước mini_batch_size, trừ batch cuối cùng
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        start = k * mini_batch_size
        end = start + mini_batch_size
        mini_batch_X = shuffled_X[start : end, :, :, :]
        mini_batch_Y = shuffled_Y[start : end, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Chia mini_batch cuối cùng (nếu mini_batch cuối < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T # np.eye(C): tạo ra ma trận đơn vị vuông cấp C
    # Y.reshape(-1): biến đổi ma trận Y thành mảng 1 chiều (m,)
    # Code trên lấy ra các vecto trong ma trận có vị trí số 1 tương ứng với giá trị trong mảng 1 chiều
    return Y

def forward_propagation_for_predict(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters['W1'])
    b1 = tf.convert_to_tensor(parameters['b1'])
    W2 = tf.convert_to_tensor(parameters['W2'])
    b2 = tf.convert_to_tensor(parameters['b2'])
    W3 = tf.convert_to_tensor(parameters['W3'])
    b3 = tf.convert_to_tensor(parameters['b3'])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3) # Tìm ra vị trí chứa giá trị lớn nhất của từng vector theo trục tung

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction
