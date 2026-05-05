import numpy as np
from scipy.stats import norm
from scipy.special import ive
from scipy.integrate import quad
from scipy.optimize import minimize, NonlinearConstraint

# =============================================================================
# --- 1. Market Data & Setup ---
# =============================================================================

# Current spot price of the underlying asset (e.g. SPY or similar equity index)
S0_mkt = 394.118

# Risk-free interest rate (annualized, continuously compounded)
r = 0.05

# Implied Black-Scholes volatility for the target call option
sigma_bs = 0.04635

# Maturity of the call option (3 months = 0.25 years)
T_call = 0.25

# Strike price set at-the-money (equals current spot)
K = S0_mkt

# Market futures prices for maturities T = 0.25, 1.25, ..., 9.25 years (k = 0 to 9)
F_mkt = np.array([17.1981, 18.7223, 19.4360, 19.7445, 19.8833,
                  19.9623, 19.9703, 20.0066, 20.0019, 20.0031])

# Corresponding futures maturities: T_k = 0.25 + k for k in {0, ..., 9}
T_k = 0.25 + np.arange(10)

# Black-Scholes call pricing formula used to compute the market target price
def bs_call(S, K, T, r, sigma):
    # Standard d1 and d2 terms from the Black-Scholes formula
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Discounted expected payoff under risk-neutral measure
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# Target call price computed from market implied volatility
Call_mkt = bs_call(S0_mkt, K, T_call, r, sigma_bs)

# =============================================================================
# --- 2. Model Functions ---
# =============================================================================

# Model-implied spot price S0 as a function of the CIR model parameters.
# Derived from the long-run equilibrium: S0 = C*/r + (C0 - C*)/(r + lambda)
def model_S0(lam, C_star, C0):
    return C_star/r + (C0 - C_star)/(r + lam)

# Model-implied futures prices at maturities T_k = 0.25 + k.
# The convenience yield (or dividend-like process) mean-reverts to C* at rate lambda.
def model_F_k(lam, C_star, C0, k_arr):
    return C_star + (C0 - C_star)/lam * np.exp(-lam * (0.25 + k_arr)) * (1 - np.exp(-lam))

# Model-implied call option price via numerical integration over the CIR density.
# The underlying follows a CIR-like process with mean-reversion speed lambda,
# long-run mean C*, initial value C0, and vol-of-vol phi.
def model_call(lam, C_star, C0, phi, T, K_strike):
    # Order parameter of the modified Bessel function (related to the Feller ratio)
    q = (2 * lam * C_star) / (phi**2) - 1

    # Scaling constant that appears in the non-central chi-squared transition density
    a = (2 * lam) / (phi**2 * (1 - np.exp(-lam * T)))

    # Linear mapping from asset price s back to the state variable g(s)
    def g(s):
        return C_star + (r + lam) * (s - C_star/r)

    # Transition density of the CIR process, expressed in terms of the asset price s.
    # Uses a numerically stable factorization to prevent overflow in the exponential.
    def density(s):
        gs = g(s)

        # Discard values outside the support of the CIR process
        if gs <= 0: return 0.0

        # Intermediate terms for the stable exponent decomposition
        sqrt_gs = np.sqrt(gs)
        sqrt_c0_exp = np.sqrt(C0 * np.exp(-lam * T))

        # Factored exponent: avoids numerical overflow by completing the square
        stable_exponent = -a * (sqrt_gs - sqrt_c0_exp)**2

        # Jacobian term from the change of variable (from CIR state to asset price)
        term1 = (r + lam) * a * np.exp(stable_exponent)

        # Power-law term from the Bessel density, scaled by the ratio gs/C0
        term2 = ((np.exp(lam * T) * gs) / C0) ** (q / 2)

        # Argument of the modified Bessel function
        z = 2 * a * np.sqrt(np.exp(-lam * T) * gs * C0)

        # Scaled modified Bessel function of the first kind: ive(q, z) = e^(-z) * I_q(z).
        # The e^(-z) factor is already absorbed into the stable_exponent above.
        term3 = ive(q, z)

        return term1 * term2 * term3

    # Lower bound on s imposed by the CIR support (gs > 0 requires s > C*/(r*(r+lam)))
    S_min = max((lam * C_star)/(r * (r + lam)), 0.01)

    # Integration starts at max(K, S_min) since payoff is zero below the strike
    lower_bound = max(K_strike, S_min)

    # Numerically integrate the discounted expected payoff: E[max(S-K, 0)] under risk-neutral measure
    integral, _ = quad(lambda s: (s - K_strike) * density(s), lower_bound, lower_bound + 200, limit=100)

    # Discount back to present value
    return np.exp(-r * T) * integral

# =============================================================================
# --- 3. Objective Function (SSE) ---
# =============================================================================

# Sum of squared errors across all calibration targets:
# spot price S0, the 10 futures prices F_k, and the call option price.
def objective(params):
    lam, C_star, C0, phi = params

    # Compute model-implied values for current parameter vector
    S0_mod = model_S0(lam, C_star, C0)
    F_mod = model_F_k(lam, C_star, C0, np.arange(10))
    Call_mod = model_call(lam, C_star, C0, phi, T_call, K)

    # SSE penalizes deviations from market observations equally across targets
    sse = (S0_mod - S0_mkt)**2 + np.sum((F_mod - F_mkt)**2) + (Call_mod - Call_mkt)**2
    return sse

# =============================================================================
# --- 4. Optimization ---
# =============================================================================

# Initial parameter guess: [lambda, C_star, C0, phi]
# Based on an analytical approximation of the long-run futures level
x0 = [0.78, 20.0, 15.12, 1.0]

# Parameter bounds: all four parameters must remain strictly positive
bounds = [(0.01, 5.0), (0.01, 50.0), (0.01, 50.0), (0.01, 5.0)]

# Feller condition: ensures the CIR process never hits zero (2*lambda*C* > phi^2)
def feller_constraint(params):
    return 2 * params[0] * params[1] - params[3]**2

# Wrap the Feller condition as a nonlinear inequality constraint for the optimizer
nlc = NonlinearConstraint(feller_constraint, 0.001, np.inf)

print("Starting Calibration...")

# Run constrained nonlinear least-squares optimization via SLSQP
result = minimize(objective, x0, bounds=bounds, constraints=nlc, method='SLSQP', options={'disp': True})

print("\n--- Calibration Results ---")
print(f"Lambda: {result.x[0]:.4f}")
print(f"C_star: {result.x[1]:.4f}")
print(f"C_0:    {result.x[2]:.4f}")
print(f"Phi:    {result.x[3]:.4f}")

# =============================================================================
# --- 5. Question 13: Hedge Ratio Calculation ---
# =============================================================================

# Extract calibrated parameters from the optimization result
lam_opt, C_star_opt, C0_opt, phi_opt = result.x

# Small perturbation used in the central finite difference approximation of delta
epsilon = 0.01

# Helper: re-price the 1Y call after shifting the spot price by delta_S.
# To remain model-consistent, C0 must be re-derived from the shifted S0
# using the inverse of the S0 pricing formula: C0 = C* + (S - C*/r) * (r + lambda)
def price_call_shifted_S(S_shift):
    C0_shift = C_star_opt + (S_shift - C_star_opt/r) * (r + lam_opt)
    return model_call(lam_opt, C_star_opt, C0_shift, phi_opt, 1.0, K)

# Central finite difference: delta = (V(S+eps) - V(S-eps)) / (2*eps)
V_up = price_call_shifted_S(S0_mkt + epsilon)
V_dn = price_call_shifted_S(S0_mkt - epsilon)
delta_call_1Y = (V_up - V_dn) / (2 * epsilon)

# Number of 2Y futures contracts needed to delta-hedge the 1Y call.
# The factor exp(2*lambda) converts the delta from spot to futures sensitivity.
n_F = delta_call_1Y * np.exp(2 * lam_opt)

print(f"\nInitial Number of 2Y Futures (n_F): {n_F:.4f}")