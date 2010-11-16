"""
A fast algorithm for the minimum covariance determinant estimator.

Author: Virgile Fritsch, 2010
        implementing a method by Rousseeuw & Van Driessen described in
        (A Fast Algorithm for the Minimum Covariance Determinant Estimator,
        1999, American Statistical Association and the American Society
        for Quality, TECHNOMETRICS)

"""
from joblib import Parallel, delayed
import numpy as np
from scipy import linalg

VERBOSE = False
DEBUG = False

def c_step_from_estimates(data, h, T, S, nb_iter=30, previous_detS=None):
    """
    
    """
    detS = linalg.det(S)
    # return optimal values if det(S) == 0...
    if detS == 0. or detS == previous_detS:
        if VERBOSE:
            print 'Optimal couple (T,S) found before ending iterations'
        return T, S, detS, nb_iter
    # ...compute a new couple of estimate otherwise
    else:
        centered_data = data - T
        dist = np.sqrt(
            (np.dot(centered_data,linalg.inv(S)) * centered_data).sum(1))
        new_H = np.argsort(dist)[:h]
    
    return c_step(data, new_H, nb_iter=nb_iter-1, previous_detS=detS)


def c_step(data, H, nb_iter=30, previous_detS=None):
    """

    """
    if len(H.shape) != 1:
        raise ValueError('H has a bad shape')
    h = H.size
    
    # robust location estimate
    T = (1./h) * data[H,:].sum(0)
    # robust covariance estimate
    centered_subsample = data[H,:] - T
    S = (1./h) * np.dot(centered_subsample.T, centered_subsample)
    
    # we stop if we have reached the maximum iterations number
    if nb_iter == 0:
        if VERBOSE:
            print 'Maximum number of iterations reached'
        result = T, S, linalg.det(S), 0
    else:
        result = c_step_from_estimates(
            data, h, T, S, nb_iter=nb_iter, previous_detS=previous_detS)
    
    return result


def run_fast_mcd(data, h, nb_trials, select=10, nb_iter=2):
    """
    
    """
    n_sub = data.shape[0]
    p = data.shape[1]
    all_Tsub = []
    all_Ssub = []
    detSsub = []
    """
    for t in range(nb_trials):
        permutation = np.random.permutation(n_sub)
        Tsub_temp, Ssub_temp, detSsub_temp, iteration = c_step(
            data, permutation[:h], nb_iter=nb_iter)
        all_Tsub.append(Tsub_temp)
        all_Ssub.append(Ssub_temp)
        detSsub.append(detSsub_temp)
    """
    
    all_Tsub, all_Ssub, detSsub, _ = zip(
        *[c_step(data, np.random.permutation(n_sub)[:h],nb_iter=nb_iter) \
         for t in range(nb_trials)])
    
    best_T = []
    best_S = []
    for i in np.argsort(detSsub)[:select]:
        best_T.append(all_Tsub[i])
        best_S.append(all_Ssub[i])
    
    return best_T, best_S


def fast_mcd(data):
    """
    
    """
    n = data.shape[0]
    p = data.shape[1]
    h = np.ceil((n+p+1)/2.)
    if n > 600:
        # split the set in subsets of size ~ 300
        nb_subsets = int(n / 300)
        n_sub = n / float(nb_subsets)
        # perform a total of 500 trials, select 10 best (T,S) for each subset
        nb_best = 10
        nb_trials = int(500 / nb_subsets)
        h_subsets = np.ceil(n_sub*(n/h))
        permutation = np.random.permutation(n)
        all_best_T, all_best_S = zip(*Parallel(n_jobs=-1, verbose=0)(
            delayed(run_fast_mcd)(
            data[permutation[np.arange(i*n_sub, (i+1)*n_sub, dtype=int),:]],
            h_subsets, nb_trials, select=nb_best) \
            for i in range(nb_subsets)))
        all_best_T = reduce(list.__add__, list(all_best_T))
        all_best_S = reduce(list.__add__, list(all_best_S))
        # pool the subsets into a merged set (possibly the full dataset)
        n_merged = min(1500,n)
        merged_subset = data[np.random.permutation(n)[:n_merged],:]
        h_merged = np.ceil(n_merged*(n/h))
        T_merged, S_merged, detS_merged, _ = zip(*Parallel(n_jobs=-1)(
            delayed(c_step_from_estimates)(
            merged_subset, h_merged, all_best_T[i], all_best_S[i]) \
            for i in range(len(all_best_T))))
                                          
        if n < 1500:
            # get the best couple (T,S)
            result_index = np.argmin(detS_merged)
            T = T_merged[result_index]
            S = S_merged[result_index]
        else:
            # find the 10 best couple (T,S) on the merged set
            nb_best_merged = 10
            best_merged_indices = np.argsort(detS_merged)
            T_best_merged = []
            S_best_merged = []
            for i in range(best_merged_indices):
                T_best_merged.append(T_merged[i])
                S_best_merged.append(S_merged[i])
            # select the best couple on the full dataset amongst the 10
            T_full, S_full, detS_full, _ = zip(*Parallel(n_jobs=-1)(
                delayed(c_step_from_estimates)(
                data, h, T_best_merged[i], S_best_merged[i]) \
                for i in range(nb_best_merged)))
            result_index = np.argmin(detS_full)
            T = T_full[result_index]
            S = S_full[result_index]
    else:
        # find the 10 best couple (T,S) considering two iterations
        nb_trials = 500
        nb_best = 10
        T_best, S_best = run_fast_mcd(data, h, nb_trials, select=nb_best)
        # select the best couple on the full dataset amongst the 10
        T_full, S_full, detS_full, _ = zip(*Parallel(n_jobs=-1)(
            delayed(c_step_from_estimates)(
            data, h, T_best[i], S_best[i]) for i in range(nb_best_merged)))
        result_index = np.argmin(detS_full)
        T = T_full[result_index,:]
        S = S_full[result_index,:,:]
    
    return T, S
