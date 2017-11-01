import numpy as np
import pickle
from itertools import compress
from scipy.stats import uniform as unif
import random
import pickle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


with open("rolls_history.p", "rb") as f:
    rolls, die, rolls_onehot = pickle.load(f)

obsModel = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
            [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]]

transmat = [[0.95, 0.05],
            [0.1, 0.90]]

pi = [.5, .5]


class HMM:
    def __init__(self, pi=pi, A=transmat, B=obsModel, evd=rolls_onehot):
        self.pi = pi
        self.A = A
        self.B = B
        self.evd = evd
        self.N = len(self.pi)
        self.T, self.M = self.evd.shape
        self.Z = np.zeros(self.T)
        self.alpha = np.zeros((self.T, self.N))
        self.beta = np.zeros((self.T, self.N))

    def normalize(self, a):
        return a / np.sum(a)


    def find_B_w_O(self, t):
        result = []
        for i in range(self.N):
            result.append(list(compress(self.B[i], self.evd[t]))[0])
        return result

    def forwards(self):
        # p(q(t)|O(1), ...., O(t))
        unnormalized = np.multiply(self.pi, self.find_B_w_O(0))
        self.alpha[0] = self.normalize(unnormalized)
        self.Z[0] = np.sum(unnormalized)
        logZ = np.log(self.Z[0] + 1e-7)

        for t in range(1, self.T):
            unnormalized = np.multiply(np.dot(self.alpha[t-1], self.A), self.find_B_w_O(t))
            self.alpha[t] = self.normalize(unnormalized)
            self.Z[t] = np.sum(unnormalized)
            logZ += np.log(self.Z[t] + 1e-7)

        return self.alpha, self.Z, logZ


    def backwards(self):
        self.beta[-1] = np.ones(self.N)
        for t in range(self.T - 2, -1, -1):
            self.beta[t] = self.normalize(np.dot(np.transpose(self.A),
                                  np.multiply(self.find_B_w_O(t + 1), self.beta[t + 1])))
        return self.beta


    def smoothing(self):
        # gamma(i, t) = p(S(t) = i|O(1:T))
        self.alpha, _, logZ = self.forwards()
        self.beta = self.backwards()
        gamma = np.multiply(self.alpha, self.beta)

        # normalize gamma to make the rowsum equal to 1
        _sum = gamma.sum(1)
        for n in range(self.N):
            gamma[:, n] = gamma[:, n]/ _sum
        return gamma




    def generate_posterior(self):
        return

