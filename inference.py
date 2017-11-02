import numpy as np
from itertools import compress
import pickle


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
        self.A = np.array(A)
        self.B = B
        self.evd = evd
        self.N = len(self.pi)
        self.T, self.M = self.evd.shape
        self.Z = np.zeros(self.T, dtype=np.int)
        self.alpha = np.zeros((self.T, self.N))
        self.beta = np.zeros((self.T, self.N))
        self.delta = np.zeros((self.T, self.N))

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


    def viterbi(self):

        # Initialization
        delta = np.zeros((self.T, self.N))
        psi = np.zeros((self.T, self.N), dtype=np.int)
        optimal_path = np.zeros(self.T, dtype=np.int)

        delta[0] = np.multiply(self.pi, self.find_B_w_O(0))

        # Induction
        for t in range(1, self.T):
            for n in range(self.N):
                delta[t, n] = np.max(delta[t-1] * self.A[:, n]) * self.find_B_w_O(t)[n]
                psi[t, n] = np.argmax(delta[t-1] * self.A[:, n])

        # Termination
        p_star = np.max(self.delta[-1])
        optimal_path[-1] = np.argmax(delta[-1])

        # Backtracking the path
        for t in range(self.T-2, -1, -1):
            optimal_path[t] = psi[t + 1, optimal_path[t + 1]]

        return p_star, optimal_path



    def generate_posterior(self):
        return


























