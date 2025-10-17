import flavio
import numpy as np
import matplotlib.pyplot as plt

# Observable name for CP asymmetry in B0->K*0 mu+ mu-
obs_name = 'ACP(B+->Kmumu)'

# q2 points (GeV^2)
q2_points = np.linspace(0.0, 22.0, 100)
q2_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12.5, 15, 17, 19, 23]

acp_values = []
for q2 in q2_points:
    acp = flavio.sm_prediction(obs_name, q2=q2)
    acp_values.append(acp)

# Plot
plt.plot(q2_points, acp_values)
plt.xlabel(r'$q^2$ [GeV$^2$]')
plt.ylabel(r'$A_{CP}(B^0 \to K^{*0} \mu^+ \mu^-)$')
plt.title('SM Prediction of Differential CP Asymmetry')
plt.grid(True)
plt.show()