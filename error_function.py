import numpy as np
import matplotlib.pyplot as plt

#output
V0_exp=
V0_sim=

def error_function(vexp, vsim):
    return np.sum(np.abs(vexp-vsim))**2

perm=[
errors=[]

for p in perm:
    err = error_function(V0_exp, V0_sim)
    errors.append(err)
    
plt.plot(perm, errors)
plt.xlabel("Permeability") #kr
plt.ylabel("Error")
plt.title("Error function")
plt.grid()
plt.show()
    
