import numpy as np

def gibbs_sampling(Phi, S, D, Values, T):
    """Gibbs' sampling from a discrete undirected model.

    Input:  Phi     - list of C functions.
            S       - list of C arrays which represent the domians for the functions 
                      in Phi. For example, S[1] = [1, 2] means that the function 
                      Phi[1] depends on the variables 1 and 2.
            D       - dimensionality of one sample, also the number of variables in 
                      the joint distribution.
            Values  - each of the D variables can take K values.
            T       - number of samples to generate.
            
    Output: Samples - D * T matrix which contains the sample in its columns. A sample
                      does not contain directly the values for the variables. Instead,
                      it contains indices for values in the vector Values.
    """
    Samples = np.ones((D, T), dtype=int)
    for t in range(1, T):
        Samples[:, t] = _gibbs_sampling_one(Samples[:, t - 1], Phi, S, D, Values)
    return Samples

def _gibbs_sampling_one(sample_start, Phi, S, D, Values):
    K = Values.size
    C = len(Phi)
    sample = sample_start
    for d in range(D):
        p_lambda = np.zeros(K)
        for k in range(K):
            sample[d] = k
            # Compute unnormalized marginal probability.
            p_lambda[k] = 1
            for c in range(C):
                if d in S[c]:
                    args = Values[sample[S[c]]]
                    p_lambda[k] *= Phi[c](args)
        p_lambda = p_lambda / np.sum(p_lambda)
        sample[d] = np.random.choice(np.arange(K), 1, replace=True, p=p_lambda)
    return sample