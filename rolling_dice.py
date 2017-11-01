from scipy.stats import uniform as unif
import random
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

random.seed(42)

two_die = ['Fair', 'Loaded']
support = [1, 2, 3, 4, 5, 6]
obsModel = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
            [1/10, 1/10, 1/10, 1/10, 1/10, 5/10]]

transmat = [[0.95, 0.05],
            [0.1, 0.90]]

pi = [.5, .5]
T = 300

def rolling_dice(n=300):

    die = [ ]
    rolls = [ ]

    # Decide starting dice
    start_rv = unif.rvs()
    if start_rv <= pi[ 0 ]:
        dice = 0
    else:
        dice = 1

    die.append(two_die[dice])

    for i in range(n):
        face = random.choices(support, obsModel[ dice ])[0]
        rolls.append(face)
        trans_rv = unif.rvs()
        if trans_rv <= transmat[dice][dice]:
            dice = dice
        else:
            dice ^= 1

        if i < n - 1:
            die.append(two_die[dice])

    rolls = np.array(rolls)
    return rolls, die

rolls, die = rolling_dice(T)
enc = OneHotEncoder()
rolls_onehot = enc.fit_transform(rolls.reshape(-1, 1)).toarray()


with open("rolls_history.p", "wb") as f:
    pickle.dump((rolls, die, rolls_onehot), f)