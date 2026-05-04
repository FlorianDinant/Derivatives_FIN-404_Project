import numpy as np
from scipy.stats import norm
from scipy.special import ive
from scipy.integrate import quad
from scipy.optimize import minimize, NonlinearConstraint

# --- 1. Market Data & Setup ---
S0_mkt = 394.118
r = 0.05
sigma_bs = 0.04635
T_call = 0.25
K = S0_mkt

# Futures Data (k=0 to 9)
F_mkt = np.array([17.1981, 18.7223, 19.4360, 19.7445, 19.8833, 
                  19.9623, 19.9703, 20.0066, 20.0019, 20.0031])
T_k = 0.25 + np.arange(10)

# Black-Scholes Pricer for Target
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

Call_mkt = bs_call(S0_mkt, K, T_call, r, sigma_bs)

# --- 2. Model Functions ---
def model_S0(lam, C_star, C0):
    return C_star/r + (C0 - C_star)/(r + lam)

def model_F_k(lam, C_star, C0, k_arr):
    return C_star + (C0 - C_star)/lam * np.exp(-lam * (0.25 + k_arr)) * (1 - np.exp(-lam))

def model_call(lam, C_star, C0, phi, T, K_strike):
    q = (2 * lam * C_star) / (phi**2) - 1
    a = (2 * lam) / (phi**2 * (1 - np.exp(-lam * T)))
    
    def g(s):
        return C_star + (r + lam) * (s - C_star/r)
    
    def density(s):
        gs = g(s)
        # Protection aux bornes
        if gs <= 0: return 0.0
        
        # L'astuce numérique de factorisation de l'exposant pour éviter l'Overflow
        sqrt_gs = np.sqrt(gs)
        sqrt_c0_exp = np.sqrt(C0 * np.exp(-lam * T))
        
        # Exposant ultra-stable grâce à l'identité remarquable
        stable_exponent = -a * (sqrt_gs - sqrt_c0_exp)**2
        
        term1 = (r + lam) * a * np.exp(stable_exponent)
        term2 = ((np.exp(lam * T) * gs) / C0) ** (q / 2)
        
        z = 2 * a * np.sqrt(np.exp(-lam * T) * gs * C0)
        term3 = ive(q, z) # ive contient déjà le e^(-z) qui a été factorisé au-dessus
        
        return term1 * term2 * term3

    S_min = max((lam * C_star)/(r * (r + lam)), 0.01)
    lower_bound = max(K_strike, S_min)
    
    # Integrate (s - K) * f(s)
    integral, _ = quad(lambda s: (s - K_strike) * density(s), lower_bound, lower_bound + 200, limit=100)
    return np.exp(-r * T) * integral

# --- 3. Objective Function (SSE) ---
def objective(params):
    lam, C_star, C0, phi = params
    
    S0_mod = model_S0(lam, C_star, C0)
    F_mod = model_F_k(lam, C_star, C0, np.arange(10))
    Call_mod = model_call(lam, C_star, C0, phi, T_call, K)
    
    sse = (S0_mod - S0_mkt)**2 + np.sum((F_mod - F_mkt)**2) + (Call_mod - Call_mkt)**2
    return sse

# --- 4. Optimization ---
# Initial Guess based on analytical approximation
x0 = [0.78, 20.0, 15.12, 1.0]

# Bounds: All strictly positive
bounds = [(0.01, 5.0), (0.01, 50.0), (0.01, 50.0), (0.01, 5.0)]

# Constraint: Feller Condition (2 * lam * C_star - phi^2 > 0)
def feller_constraint(params):
    return 2 * params[0] * params[1] - params[3]**2

nlc = NonlinearConstraint(feller_constraint, 0.001, np.inf)

print("Starting Calibration...")
result = minimize(objective, x0, bounds=bounds, constraints=nlc, method='SLSQP', options={'disp': True})

print("\n--- Calibration Results ---")
print(f"Lambda: {result.x[0]:.4f}")
print(f"C_star: {result.x[1]:.4f}")
print(f"C_0:    {result.x[2]:.4f}")
print(f"Phi:    {result.x[3]:.4f}")

# --- 5. Question 13: Hedge Ratio Calculation ---
lam_opt, C_star_opt, C0_opt, phi_opt = result.x

# Finite difference for Delta Call (1Y maturity)
epsilon = 0.01
# Trick: to change S0, we must slightly shift C0 to keep the model consistent
def price_call_shifted_S(S_shift):
    C0_shift = C_star_opt + (S_shift - C_star_opt/r) * (r + lam_opt)
    return model_call(lam_opt, C_star_opt, C0_shift, phi_opt, 1.0, K)

V_up = price_call_shifted_S(S0_mkt + epsilon)
V_dn = price_call_shifted_S(S0_mkt - epsilon)

delta_call_1Y = (V_up - V_dn) / (2 * epsilon)
n_F = delta_call_1Y * np.exp(2 * lam_opt)

print(f"\nInitial Number of 2Y Futures (n_F): {n_F:.4f}")