"""

Author: Virgile Fritsch, 2010

"""
import numpy as np
from scipy import linalg
from scipy.stats import chi2
import matplotlib.pyplot as plt

from fast_mcd import fast_mcd

PLOT = True

def generate_gaussian(n, p, means, vars):
    """
    
    """
    ### Generate data
    data = np.random.randn(n, p)
    data = np.dot(data, np.diag(vars)) + means

    return data

def generate_nongaussian(n, p, means_1, means_2, vars_1, vars_2):
    """
    
    """
    ### Generate data
    data = np.random.randn(n, p)
    data = np.dot(data, np.diag(vars_1)) + means_1
    data_add = np.random.randn(n,p)
    data_add = np.dot(data_add, np.diag(vars_2)) + means_2
    data += data_add

    return data

def generate_gaussian_outliers(n, p, means, vars,
                               nb_outliers, means_outliers, vars_outliers):
    """

    """
    ### Generate gaussian data
    data = np.random.randn(n+nb_outliers, p)
    data[:n,:] = np.dot(data[:n,:], np.diag(vars)) + means

    # add outliers
    data[n:,:] = np.dot(data[n:,:], np.diag(vars_outliers)) + means_outliers

    return data


def test_fast_mcd(data):
    """

    """
    n = data.shape[0]
    p = data.shape[1]
    
    ### Naive location and scatter estimates
    location = data.mean(0)
    covariance = np.cov(data.T)
    # invert the covariance matrix
    try:
        inv_sigma = linalg.inv(robust_covariance)
    except:
        u, s, vh = linalg.svd(covariance)
        inv_s = (1. / s) * \
                ((np.cumsum(s) < np.sum(s) * .95) | ([True]+[False]*(len(s)-1)))
        inv_sigma = np.dot(np.dot(vh.T, np.diag(inv_s)), u.T)
    # get distribution of data's Mahalanobis distances
    Y = data - location
    R = np.sqrt((np.dot(Y, inv_sigma) * Y).sum(1))
    # estimate the density with a gaussian kernel
    nonnan_subjects_arg = np.where(~np.isnan(R))[0]
    R = R[nonnan_subjects_arg]
    x1 = np.arange(0., 1.2*np.amax(R), 0.0012*np.amax(R))
    n = R.size
    sigma = 1.05 * np.std(R) * n**(-0.2)
    kernel_arg = (np.tile(x1, (n,1)).T - R) / sigma
    fh = ((1/np.sqrt(2*np.pi)) * np.exp(-0.5*kernel_arg**2)).sum(1) / (n*sigma)
    # plot the distribution
    if PLOT:
        plt.figure()
        plt.plot(x1, fh, color='blue')
    # Khi-2 distribution
    diff_scale = np.sqrt(R.var() / float(chi2.stats(p, moments='v')))
    diff_loc = R.mean() - float(chi2.stats(p, scale=diff_scale, moments='m'))
    template = chi2(p, loc=diff_loc, scale=diff_scale)
    if PLOT:
        plt.plot(x1, template.pdf(x1), linestyle='--', color='blue')
    mse_naive = ((fh - template.pdf(x1))**2).mean()
    imse_naive = 0.5 * ((fh - template.pdf(x1))**2).sum() * (x1[1] - x1[0])
    if PLOT:
        print "MSE (naive case) =", mse_naive
        print "IMSE (naive case) =", imse_naive
    
    ### Robust location and scatter estimates
    robust_location, robust_covariance = fast_mcd(data)
    try:
        inv_sigma = linalg.inv(robust_covariance)
    except:
        u, s, vh = linalg.svd(robust_covariance)
        inv_s = (1. / s) * \
                ((np.cumsum(s) < np.sum(s) * .95) | ([True]+[False]*(len(s)-1)))
        inv_sigma = np.dot(np.dot(vh.T, np.diag(inv_s)), u.T)
    # get distribution of data's Mahalanobis distances
    Y = data - robust_location
    R = np.sqrt((np.dot(Y, inv_sigma) * Y).sum(1))
    # estimate the density with a gaussian kernel
    nonnan_subjects_arg = np.where(~np.isnan(R))[0]
    R = R[nonnan_subjects_arg]
    x2 = np.arange(0., 1.2*np.amax(R), 0.0012*np.amax(R))
    n = R.size
    sigma = 1.05 * np.std(R) * n**(-0.2)
    kernel_arg = (np.tile(x2, (n,1)).T - R) / sigma
    fh = ((1/np.sqrt(2*np.pi)) * np.exp(-0.5*kernel_arg**2)).sum(1) / (n*sigma)
    # plot the distribution
    if PLOT:
        plt.plot(x2, fh, color='green')
    # Khi-2 distribution
    diff_scale = np.sqrt(R.var() / float(chi2.stats(p, moments='v')))
    diff_loc = R.mean() - float(chi2.stats(p, scale=diff_scale, moments='m'))
    template = chi2(p, loc=diff_loc, scale=diff_scale)
    if PLOT:
        plt.plot(x2, template.pdf(x2), linestyle='--', color='green')
    mse_robust = ((fh - template.pdf(x2))**2).mean()
    imse_robust = 0.5 * ((fh - template.pdf(x2))**2).sum() * (x2[1] - x2[0])
    if PLOT:
        print "MSE (robust case) =", mse_robust
        print "IMSE (robust case) =", imse_robust
        plt.legend(('empirical distribution (naive)', 'chi-2 (naive)',
                    'empirical distribution (robust)', 'chi-2 (robust)'),
                   loc='upper center', bbox_to_anchor=(0.5, 0.))
        plt.show()
    
    return mse_naive, mse_robust, imse_naive, imse_robust


if __name__ == "__main__":
    PLOT = False
    p_min = 100
    p_max = 100
    nb_loops = 10
    accu_mse_naive = np.zeros(p_max-p_min+1)
    accu_mse_robust = np.zeros(p_max-p_min+1)
    accu_imse_naive = np.zeros(p_max-p_min+1)
    accu_imse_robust = np.zeros(p_max-p_min+1)
    for p in range(p_min,p_max+1):
        print p
        for i in range(nb_loops):
            #mse_naive, mse_robust, imse_naive, imse_robust = test_fast_mcd(
            #   generate_gaussian(700, p, np.tile([0], p), np.tile([1], p)))
            mse_naive, mse_robust, imse_naive, imse_robust = test_fast_mcd(
                generate_gaussian(1000, p, np.arange(1,p+1), np.arange(1,p+1)))
            accu_mse_naive[p-p_min] += mse_naive
            accu_mse_robust[p-p_min] += mse_robust
            accu_imse_naive[p-p_min] += imse_naive
            accu_imse_robust[p-p_min] += imse_robust
    accu_mse_naive /= float(nb_loops)
    accu_mse_robust /= float(nb_loops)
    accu_imse_naive /= float(nb_loops)
    accu_imse_robust /= float(nb_loops)

    # MSE plot
    #plt.figure()
    #plt.plot(np.arange(2,p_max+1), accu_mse_naive)
    #plt.plot(np.arange(2,p_max+1), accu_mse_robust)
    #plt.legend(("naive MSE", "robust MSE"))
    #plt.title("naive and robust MSE")
    #plt.figure()
    #plt.plot(np.arange(2,p_max+1), accu_mse_naive-accu_mse_robust)
    #plt.title("Difference between naive and robust approach (MSE)")
    # IMSE plot
    plt.figure()
    plt.plot(np.arange(2,p_max+1), accu_imse_naive)
    plt.plot(np.arange(2,p_max+1), accu_imse_robust)
    plt.legend(("naive IMSE", "robust IMSE"))
    plt.title("naive and robust IMSE")
    plt.figure()
    plt.plot(np.arange(2,p_max+1), accu_imse_naive-accu_imse_robust)
    plt.title("Difference between naive and robust approach (IMSE)")

    plt.show()
