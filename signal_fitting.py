import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zfit

fs_data = pd.read_pickle("/Users/zifei/Desktop/C1/ProcessedData/first_selection_data.pkl")
fs_data = fs_data[(fs_data['B_invariant_mass'] < 5500) & (fs_data['B_invariant_mass'] > 5000)]
count, bins = np.histogram(fs_data['B_invariant_mass'], bins=100)

mass = zfit.Space('B_invariant_mass', (5000, 5500))

#Next we define the model parameters (an initial guess and their allowed ranges)
#For the signal we've got a gaussian, so need a mean and sigma, and of course a
#yield (or number of signal events)

mean  = zfit.Parameter('mean', 5280, 5000, 5500)
sigma = zfit.Parameter('sigma', 20,    5,  100)
signalYield = zfit.Parameter('signalYield', 1000, 0, 20000)

#Now to define the signal model
signalPDF = zfit.pdf.Gauss(obs=mass, mu=mean, sigma=sigma, extended=signalYield)


#We're using an exponential for the background model, so we have an exponent
#parameter (and a yield)
exponent  = zfit.Parameter('exponent', -0.01, -1e-1, -1e-6)
backgroundYield = zfit.Parameter('backgroundYield', 1000, 0, 20000)

#define the background model
backgroundPDF = zfit.pdf.Exponential(obs=mass, lam=exponent, extended=backgroundYield)

#The total pdf to fit to the data is the sum of the signal and background
totalPDF = zfit.pdf.SumPDF(pdfs=[signalPDF, backgroundPDF])
zdata = zfit.Data.from_pandas(fs_data, mass)
nll = zfit.loss.ExtendedUnbinnedNLL(model=totalPDF, data=zdata)

#We then ask the minimiser (Minuit here) find the parameters that minimise the
#liklehood
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)

#To compute the uncertainty on the parameters we can use Hesse. This estimates
#the uncertainties from the second gradient at the minimum. This is good when
#the likelihood is nicely parabolic, i.e. in the asymptotic limit. The coverage
#of Hesse should always be checked. There are alternatives, e.g. Minos that
#can cope with slightly asymmetric minima, but again the coverage should always
#be checked.
param_hesse = result.hesse()

#print the result, and also explicitly access the result information. This is
#important as the minimiser will always complete, but that doesn't mean it has
#converged or that it has found a sensible minimum. Minuit has a few parameters
#to tweak how it behaves, the strategy and tolerance are usually the most
#important, but sometimes it is necessary to kick the minimiser out of e.g. a
#local minimum.
print(result)
print(f"Function minimum: {result.fmin}")
print(f"Converged: {result.converged}")
print(f"Valid: {result.valid}")

bins = 50

#Going to use a "pretty" style of plotting...
y, bin_edges = np.histogram(fs_data['B_invariant_mass'], bins=bins)
binwidth = bin_edges[1] - bin_edges[0]
x    = 0.5*(bin_edges[1:] + bin_edges[:-1])
xerr = 0.5*(bin_edges[1:] - bin_edges[:-1])
yerr = np.sqrt(y)

plt.errorbar(x, y, xerr=xerr, yerr=yerr, color='black', fmt='.',label='Data')

x_plot = np.linspace(mass.lower[0], mass.upper[0], num=1000)
tot_plot = (signalYield+backgroundYield)*binwidth*totalPDF.pdf(x_plot)
plt.plot(x_plot, tot_plot, color='xkcd:blue', label='Total')
sig_plot = signalYield*binwidth*signalPDF.pdf(x_plot)
plt.plot(x_plot, sig_plot, color='xkcd:red', label=r'Signal')
bkg_plot = backgroundYield*binwidth*backgroundPDF.pdf(x_plot)
plt.plot(x_plot, bkg_plot, color='xkcd:green', label='Background')
plt.xlabel(r'B candidate mass / MeV/$c^2$')
plt.ylabel(r'Candidates / (10 MeV/$c^2)$')
plt.legend()
plt.savefig('/Users/zifei/Desktop/fit_bmeson.png', dpi=300)
plt.show()