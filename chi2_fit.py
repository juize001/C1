import numpy as np
import joblib
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt

# Load your saved pulls
all_pulls = joblib.load("binned_pulls2.pkl")

# Storage
bin_indices = []
pull_means = []
pull_sigmas = []
chi2_vals = []
chi2_ndfs = []
pvals = []
toy_sys = []

# Loop over bins
for bin_idx, result in all_pulls.items():
    pulls = np.array(result["pulls"])
    pulls = pulls[np.isfinite(pulls)]  # safety
    
    mean = np.mean(pulls)
    sigma = np.std(pulls)
    
    chi2 = np.sum(pulls**2)
    ndf = len(pulls)
    chi2_ndf = chi2 / ndf
    pval = 1 - chi2_dist.cdf(chi2, df=ndf)
    
    bin_indices.append(bin_idx)
    pull_means.append(mean)
    pull_sigmas.append(sigma)
    chi2_vals.append(chi2)
    chi2_ndfs.append(chi2_ndf)
    pvals.append(pval)
    toy_sys.append(sigma / np.sqrt(200))

    print(f"Q² bin {bin_idx}: mean={mean:.3f}, sigma={sigma:.3f}, chi2/ndf={chi2_ndf:.3f}, pval={pval:.3f}, toysys={toy_sys[-1]:.3f}")

# === PLOTTING ===

# 1) Pull mean
plt.figure()
plt.axhline(0, linestyle="--")
plt.scatter(bin_indices, pull_means)
plt.title("Pull Mean per Q² Bin")
plt.xlabel("Q² bin")
plt.ylabel("Mean(pull)")
plt.show()

# 2) Pull sigma
plt.figure()
plt.axhline(1, linestyle="--")
plt.scatter(bin_indices, pull_sigmas)
plt.title("Pull Sigma per Q² Bin")
plt.xlabel("Q² bin")
plt.ylabel("Sigma(pull)")
plt.show()

# 3) Chi2/ndf
plt.figure()
plt.axhline(1, linestyle="--")
plt.scatter(bin_indices, chi2_ndfs)
plt.title("Chi2/ndf per Q² Bin")
plt.xlabel("Q² bin")
plt.ylabel("Chi2/ndf")
plt.show()
