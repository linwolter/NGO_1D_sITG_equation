import numpy as np
from scipy.linalg import eig, solve
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set parameters
N_SAMPLES = 10000  # Number of training samples
Nv = 32          # Velocity grid points
Nt = 64      # Time steps
t_max = 10       # Final simulation time
vmax, vmin = 4, -4

# Parameter ranges
ranges = {
    'omega_n':  [0.0, 20.0],
    'omega_Ti': [10.0, 60.0],
    'ky':       [0.1, 1.0],
    'kz':       [1.0, 4.0],
    'c_i':      [0.01, 1.0]
}

# static precomputation
v = np.linspace(vmin, vmax, Nv)
t = np.linspace(0, t_max, Nt)
dv = (vmax - vmin) / (Nv - 1)
v_matrix = np.diag(v)
F0 = (1.0 / np.pi**0.5) * np.exp(-v**2)
F0_matrix = np.diag(F0)
W = np.ones((Nv, Nv)) * dv
Integral_Operator = F0_matrix @ W 

# Storage 
X_params = np.zeros((N_SAMPLES, 4))
X_initial_conditions = np.zeros((N_SAMPLES, Nv), dtype=np.complex128) 
Y_solutions = np.zeros((N_SAMPLES, Nt, Nv), dtype=np.complex128)

print(f"Generating {N_SAMPLES} Vlasov solutions...")

# Loop for all samples
for i in tqdm(range(N_SAMPLES)):
    
    # Random Parameters
    omega_n  = np.random.uniform(*ranges['omega_n'])
    omega_Ti = np.random.uniform(*ranges['omega_Ti'])
    ky       = np.random.uniform(*ranges['ky'])
    kz       = np.random.uniform(*ranges['kz'])
    X_params[i] = [omega_n, omega_Ti, ky, kz]
    
    # Vlasov Operator A
    bracket_term = omega_n + (v**2 - 0.5) * omega_Ti
    bracket_matrix = np.diag(bracket_term)
    
    A1 = -1j * kz * v_matrix
    A2 = -1j * kz * (v_matrix @ Integral_Operator)
    A3 = -1j * ky * (bracket_matrix @ Integral_Operator)
    A = A1 + A2 + A3
    
    # Solve
    evals, evecs = eig(A)
    
    # Random coefficients
    coeffs = np.random.uniform(*ranges['c_i'], size=Nv) + \
        1j * np.random.uniform(*ranges['c_i'], size=Nv)
    
    # Time evolution part, with eigenvalues
    time_evol = np.exp(np.outer(t, evals))
    
    # Summation to get solution f
    f_vt = np.einsum('tj, j, vj -> tv', time_evol, coeffs, evecs)
    
    # Store initial condition (t=0)
    X_initial_conditions[i] = f_vt[0] 
    
    # Store full time evolution
    Y_solutions[i] = f_vt

# Save the data
filename = 'vlasov_ngo_data_64x32.npz'
np.savez(filename, 
         params=X_params, 
         initial_conditions=X_initial_conditions, 
         solutions=Y_solutions)

print(f"\nDone! Saved to {filename}")
print(f"Parameter Shape: {X_params.shape}")
print(f"Initial Condition Shape: {X_initial_conditions.shape} (Complex)") 
print(f"Solution Shape: {Y_solutions.shape} (Complex)")

# plot a random solution to check
plt.figure()
plt.pcolormesh(v, t, Y_solutions[50].real)
plt.show()