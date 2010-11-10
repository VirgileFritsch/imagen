"""
Script to find outliers for 1 contrast, 1 atlas, using mahalanobis distance.

Author: Virgile Fritsch, 2010
        from original Benjamin Thyreau's scripts

"""
import numpy as np
from scipy import linalg
from scipy.stats import chi2
import glob
from nipy.io.imageformats import load
import matplotlib.pyplot as plt

from fast_mcd import fast_mcd

#---------------------------------------------------------------
#----- Parameters ----------------------------------------------
#---------------------------------------------------------------

#----- Database path
DB_PATH = "/volatile/imagen/data"

#----- Functional Task
FUNCTIONAL_TASK = "GCA"

#----- Contrast number
CONTRAST = 13

#----- Atlas type ("cortical", "subcortical", or "functional")
ATLAS_TYPE = "subcortical"

#---------------------------------------------------------------
#----- Routines ------------------------------------------------
#---------------------------------------------------------------

def precollect(labels, skipfirst=False):
    """

    """
    labelsvec = np.ravel(labels)
    argposlabels = np.argsort(labelsvec)
    b = labelsvec[argposlabels]
    bounds = [0] + list(np.where(b[1:]!=b[:-1])[0]+1) + [len(b)]
    if skipfirst:
        bounds = bounds[1:]

    return np.unique(b)[skipfirst:-1], argposlabels, bounds[skipfirst:]


def collect(signal, argposlabels, bounds):
    """Extracts each ROIs values

    Parameters
    ----------
    - signal: data from the nifti image.
    - argposlabels: a rearrangement of the indices so that the values
      of a given ROIs are consecutive in the flat data array.
    - bounds: location of the boundary between two ROIs values (as indices)
    
    """
    sorted_signal = signal.ravel()[argposlabels]
    
    return [sorted_signal[b:e] for b,e in zip(bounds[:-1], bounds[1:])]

"""
def enlarge_trick(img, targetimg):
    Returns data from a niftiimage,
    optionnaly broadcasting up to len(targetimg) on newaxis

    
    if img.extent == target_img.extent:
        enlarged_data = img.get_data()
    else:
        enlarged_data = as_strided(
            img.get_data(), shape=(len(targetimg.data),)+img.get_data().shape,
            strides=(0,)+img.get_data().strides)
    
    return enlarged_data
"""

#---------------------------------------------------------------
#----- Script starts -------------------------------------------
#---------------------------------------------------------------
# Subjects directory
subjects_dir = "%s/%s" %(DB_PATH, FUNCTIONAL_TASK)

### Load atlas
# get (or compute) atlas data
if ATLAS_TYPE == "cortical":
    ATLAS = "/volatile/imagen/templateEPI/cort_333_LR.nii.gz"
    atlas = load(ATLAS)
elif ATLAS_TYPE == "subcortical":
    ATLAS = "/volatile/imagen/templateEPI/subcort_333.nii.gz"
    atlas = load(ATLAS)
elif ATLAS_TYPE == "functional":
    # /!\ NOT IMPLEMENTED YET
    pass
else:
    raise Exception("Wrong atlas type.")
# load ROIs structures
labels, argposlabels, bounds = precollect(atlas.get_data(), skipfirst=True)


print "Computing the covariance matrix of ROIs means for the %s atlas" \
      %(ATLAS_TYPE)
# get all subjects data (estimables contrasts only)
filenames = sorted(
    glob.glob("%s/0000*/con_%04d.nii.gz" %(subjects_dir, CONTRAST)))
actual_files = [f for f in filenames if not "unestimable" in load(f).get_header()['descrip']]
images = [load(f) for f in actual_files]
# find unestimables contrasts
unestimable_files = sorted(set(filenames).difference(actual_files))
print "unestimable: ", unestimable_files
unestimable_subjects = [x[42:54] for x in unestimable_files]
# get ROIs mean values for each subject
allsummary = np.zeros((len(actual_files), len(labels)))
for i, img in enumerate(images):
    values = collect(img.get_data(), argposlabels, bounds)
    allsummary[i,:] = np.asarray([v[~np.isnan(v)].mean() for v in values])
    del img
# deal with the NaNs, since they confuse argsort
nonan_mask = np.isnan(allsummary).sum(1) == 0 
fallsummary = allsummary[nonan_mask,:]
# --------> /!\ fixme: find a better way to trim
# per ROI trimmed-list of subjects (10 each side)
M = np.argsort(fallsummary, 0)[10:-10]
trimmed_ind = reduce(np.intersect1d, M.T)
trimmed_allsummary = fallsummary[trimmed_ind]
del M
#trimmed_allsummary = fallsummary

# SVD decomposition of the covariance matrix
covariance = np.cov(trimmed_allsummary.T)
robust_location, robust_covariance = fast_mcd(fallsummary)
u, s, vh = linalg.svd(robust_covariance)
# --------> /!\ fixme: look at that criterion (75%)
# keep only 75% of the covariance
inv_s = (1. / s) * \
        ((np.cumsum(s) < np.sum(s) * .95) | ([True]+[False]*(len(s)-1)))
inv_sigma = np.dot(np.dot(vh.T, np.diag(inv_s)), u.T)

# --------> /!\ fixme: median ?
# compute Mahalanobis distances
Y = fallsummary - robust_location
#Y = fallsummary - np.mean(fallsummary, 0)
R = np.sqrt((np.dot(Y, inv_sigma) * Y).sum(1))
# find outliers threshold
sortedR = R[~np.isnan(R)].copy()
sortedR.sort()
qi, qe, qa = np.outer(len(sortedR), [0.25, 0.5, 0.75])[0]
bnd = (sortedR[qa] - sortedR[qi])*3 + sortedR[qe]

### Estimate the density with a gaussian kernel
nonnan_subjects_arg = np.where(~np.isnan(R))[0]
R = R[nonnan_subjects_arg]
x = np.arange(0., 1.2*np.amax(R), 0.0012*np.amax(R))
n = R.size
sigma = 1.05 * np.std(R) * n**(-0.2)
kernel_arg = (np.tile(x, (n,1)).T - R) / sigma
fh = ((1/np.sqrt(2*np.pi)) * np.exp(-0.5*kernel_arg**2)).sum(1) / (n*sigma)
# print it
plt.figure()
plt.plot(x, fh)
plt.vlines(sortedR[qe], 0, np.amax(fh))
plt.vlines(bnd, 0, np.amax(fh))
# Khi-2 distribution
p = labels.size
diff_scale = np.sqrt(R.var() / float(chi2.stats(p, moments='v')))
diff_loc = R.mean() - float(chi2.stats(p, scale=diff_scale, moments='m'))
template = chi2(p, loc=diff_loc, scale=diff_scale)
plt.plot(x, template.pdf(x), linestyle='--', color='green')
plt.show()



for i in np.where(R > bnd)[0]:
    print actual_files[nonnan_subjects_arg[i]][26:38]
