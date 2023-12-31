import numpy as np
import matplotlib.pyplot as plt

# -- 各設定値 --
n_time = 15  # 時系列の数を15に設定
n_in = 1  # 入力層のニューロン数
n_mid = 20  # 中間層のニューロン数
n_out = 1  # 出力層のニューロン数

eta = 0.001  # 学習係数
epochs = 51
batch_size = 8
interval = 5  # 経過の表示間隔

# -- 訓練データの作成 --
cos_x = np.linspace(-2*np.pi, 2*np.pi)  # -2πから2πまで
cos_y = np.cos(cos_x) + 0.1*np.random.randn(len(cos_x))  # cos関数に乱数でノイズを加える

n_sample = len(cos_x)-n_time  # サンプル数
input_data = np.zeros((n_sample, n_time, n_in))  # 入力
correct_data = np.zeros((n_sample, n_out))  # 正解
for i in range(0, n_sample):
    input_data[i] = cos_y[i:i+n_time].reshape(-1, 1)
    correct_data[i] = cos_y[i+n_time:i+n_time+1]  # 正解は入力よりも一つ後

# -- RNN層 -- 
class SimpleRNNLayer:
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)  # Xavierの初期値
        self.v = np.random.randn(n, n) / np.sqrt(n)  # Xavierの初期値
        self.b = np.zeros(n)

    def forward(self, x, y_prev):  # y_prev: 前の時刻の出力
        u = np.dot(x, self.w) + np.dot(y_prev, self.v) + self.b
        self.y = np.tanh(u)  # 出力

    def backward(self, x, y, y_prev, grad_y):
        delta = grad_y * (1 - y**2)
        self.grad_w = np.dot(x.T, delta)
        self.grad_v = np.dot(y_prev.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        self.grad_y_prev = np.dot(delta, self.v.T)

    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    def update(self, eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        self.b -= eta * self.grad_b

# -- 全結合 出力層 --
class OutputLayer:
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)  # Xavierの初期値
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u  # 恒等関数

    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T) 

    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

# 各層の初期化
rnn_layer = SimpleRNNLayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)

# 訓練
def train(x_mb, t_mb):
    y_rnn = np.zeros((len(x_mb), n_time+1, n_mid))
    y_prev = y_rnn[:, 0, :]
    for i in range(n_time):
        x = x_mb[:, i, :]
        rnn_layer.forward(x, y_prev)
        y = rnn_layer.y
        y_rnn[:, i+1, :] = y
        y_prev = y

    output_layer.forward(y)
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    rnn_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x = x_mb[:, i, :]
        y = y_rnn[:, i+1, :]
        y_prev = y_rnn[:, i, :]
        rnn_layer.backward(x, y, y_prev, grad_y)
        grad_y = rnn_layer.grad_y_prev

    rnn_layer.update(eta)
    output_layer.update(eta)

# 予測
def predict(x_mb):
    y_prev = np.zeros((len(x_mb), n_mid))
    for i in range(n_time):
        x = x_mb[:, i, :]
        rnn_layer.forward(x, y_prev)
        y = rnn_layer.y
        y_prev = y

    output_layer.forward(y)
    return output_layer.y

# 誤差を計算
def get_error(x, t):
    y = predict(x)
    return 1.0/2.0*np.sum(np.square(y - t))

error_record = []
n_batch = len(input_data) // batch_size  # 1エポックあたりのバッチ数
for i in range(epochs):
    index_random = np.arange(len(input_data))
    np.random.shuffle(index_random)  # インデックスをシャッフルする
    for j in range(n_batch):
        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x_mb = input_data[mb_index, :]
        t_mb = correct_data[mb_index, :]
        train(x_mb, t_mb)

    error = get_error(input_data, correct_data)
    error_record.append(error)

    if i % interval == 0:
        print("Epoch:", i+1, "/", epochs, "Error:", error)

plt.plot(range(1, len(error_record)+1), error_record)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
