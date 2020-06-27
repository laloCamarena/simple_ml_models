import numpy as np

class Perceptron:
    def __init__(self, w, eta, iterations):
        self.w = w
        self.eta = eta
        self.iter = iterations
        self._get_y = lambda x: 1 if np.matmul(self.w, x) >= 0 else -1

    def fit(self, X, y):
        print('fitting model')
        for x in range(self.iter):
            i = 0
            for row in X:
                row = np.append([1], row)
                self.w = self.w + ((self.eta * (y[i] - self._get_y(row))) * row)
                i += 1

    def predict(self, X):
        print('predicting stuff')
        l = []
        for x in X:
            x = np.append([1], x)
            if np.matmul(self.w, x) >= 0:
                l.append(1)
            else:
                l.append(0)
        return np.array(l)