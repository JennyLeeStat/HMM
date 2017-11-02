# Dishonest Casino - Inference in HMM
Uncertainty arises when there is incomplete information. Suppose there’s an incomplete information on a sequence of variable, but we can only observe other variables depending on those variables. In this case, we can build a Hidden Markov Model to model the rational decision that plans for the future to maximize its objective. In this post, several kinds of inference problems of discrete Hidden Markov Models are explored - filtering, smoothing, and most likely explanation
The related blog post can be found [here](https://www.jennyleestat.com/post/hmm-algorithms/).

# Toy example - dishonest casino
Let’s consider an example called “occasionally dishonest casino”. The casino choose one dice between two die. We can only observe the rolls, and have to guess which dice was used.

# Estimation results
## Filtering - forward algorithm
![Settings Window](https://github.com/JennyLeeStat/HMM/blob/master/img/filtering.png?raw=true)
## Smoothing - forward and backward
![Settings Window](https://github.com/JennyLeeStat/HMM/blob/master/img/smoothing.png)
## Most likely sequence - Viterbi decoding
![Settings Window](https://github.com/JennyLeeStat/HMM/blob/master/img/optimal_path.png?raw=true)

# Reference
- Machine Learning, a Probabilistic Perspective, K. P. Murphy(2012)
- A Tutorial on Hidden Markov Models and Selected Application in Speech Recognition, L.R. Rabiner (1989)

