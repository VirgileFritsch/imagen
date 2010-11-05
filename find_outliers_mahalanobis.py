"""
Script to find outliers for 1 contrast, 1 atlas, using mahalanobis distance.

Author: Virgile Fritsch, 2010
        from original Benjamin Thyreau's scripts

"""
import numpy as np
from scipy import linalg
import glob
from nipy.io.imageformats import load

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
ATLAS_TYPE = "cortical"

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
"""
M = np.argsort(fallsummary, 0)[10:-10]
trimmed_ind = reduce(np.intersect1d, M.T)
trimmed_allsummary = fallsummary[trimmed_ind]
del M
"""
trimmed_allsummary = fallsummary

# SVD decomposition of the covariance matrix
u, s, vh = linalg.svd(np.cov(trimmed_allsummary))
# --------> /!\ fixme: look at that criterion (75%)
# keep only 75% of the covariance
inv_s = (1. / s) * \
       ((np.cumsum(s) < np.sum(s) * .75) | ([True]+[False]*(len(s)-1)))
inv_sigma = np.dot(np.dot(vh.T, np.diag(inv_s)), u.T)

# --------> /!\ fixme: median ?
Y = fallsummary - np.median(fallsummary, 0)
R = np.sqrt(np.dot(np.dot(Y.T, inv_sigma), Y).sum(1))
R[np.isnan(R)] = 0.

sortedR = R.copy()
sortedR.sort()
qi, qe, qa = np.outer(len(R), [0.25, 0.5, 0.75]) [0]
bnd = np.abs(sortedR[qa] - sortedR[qi])*3 + sortedR[qe]
