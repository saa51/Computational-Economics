# Computational Economics

This repository contains implementations of various computational economics models, including optimal growth models, incomplete markets models, and function approximation methods. The project was developed for coursework in computational economics.

## Directory Structure

```
Computational-Economics/
├── main.py                    # Problem Set 1 entry point
├── problem_set_2.py           # Problem Set 2 entry point  
├── problem_set_3.py           # Problem Set 3 (Huggett model)
├── final_project.py           # Final project - income and wealth dynamics
├── final2.py                  # Final project utilities
│
├── Environment Modules (Models)
├── OptimalGrowth.py           # Solow growth model with AR(1) TFP
├── SOGEnv.py                  # Stochastic Optimal Growth environment
├── Huggett1996Env.py          # Huggett (1996) incomplete markets model
├── GrowthWithHabitIACEnv.py   # Growth model with habit formation & IAC
├── KrusellSmithEnv.py         # Krusell-Smith (1998) incomplete markets
├── FuncApproxEnv.py           # Environment for comparing approximation methods
│
├── Numerical Methods
├── FunctionApprox.py          # Function approximation tools
├── MarkovApprox.py            # Markov chain approximations
├── RandomProcess.py           # Random process simulation
│
├── Utilities
├── utilize.py                 # Utility functions (tables, Gini, etc.)
├── menu_cost.py               # Menu cost model
│
├── data/                      # Input data files
├── figures/                   # Generated figures
│   ├── problem_set_1/
│   ├── problem_set_2/
│   ├── problem_set_3/
│   └── final_project/
│
└── reports/                   # LaTeX reports
```

---

## Core Modules

### Economic Models

#### OptimalGrowth.py
Implements a Solow growth model with AR(1) total factor productivity (TFP) following Kopecky and Suen (2010). Features:
- Value function iteration with Chebyshev approximation
- Grid search and Euler equation methods
- Markov chain approximation of AR(1) process using Rowenhorst method
- Monte Carlo simulation for moment computation

**Key Classes:**
- `OptimalGrowthEnv`: Main environment class

**Key Methods:**
- `grid_search()`: Solve using value function iteration with grid search
- `euler_method()`: Solve using Euler equation approach
- `simulation()`: Run Monte Carlo simulation
- `compute_moments()`: Compute and compare model moments

---

#### SOGEnv.py
Stochastic Optimal Growth model environment used in Problem Set 2. Features:
- Value function iteration
- Chebyshev approximation
- Modified Policy Function Iteration (PEA)
- Endogenous Grid Method (EGM)

**Key Classes:**
- `SOGENV`: Main environment class
- `SOGVFIResult`: Results container with plotting utilities

**Key Methods:**
- `value_func_iter()`: Standard VFI
- `value_approx()`: Value function approximation
- `modified_pea()`: Modified PEA algorithm
- `endo_grid()`: Endogenous grid method
- `simulate()`: Simulate model paths

---

#### Huggett1996Env.py
Implements the Huggett (1996) overlapping generations model with incomplete markets. Features:
- Life-cycle model with borrowing constraints
- Pension system
- Stationary equilibrium computation
- Wealth distribution analysis

**Key Classes:**
- `Huggett1996Env`: Main environment class

**Key Methods:**
- `solve_stationary_equilibrium()`: Find equilibrium prices and allocations
- `gen_stationary_distribution()`: Compute stationary distribution
- `gini_index()`: Calculate Gini coefficient
- `plot_Lorenz_curve()`: Visualize inequality
- `plot_policy()`: Plot decision rules

---

#### GrowthWithHabitIACEnv.py
Growth model with habit utility and investment adjustment costs (Problem Set 1, Question 3). Features:
- Habit formation (internal and external)
- Investment adjustment costs (IAC)
- Policy function iteration (PEA)
- Steady state computation

**Key Classes:**
- `GrowthWithHabitIACEnv`: Main environment class

**Key Methods:**
- `solve_steady_state()`: Compute steady state
- `pea()`: Policy function iteration
- `value_func_approx()`: Value function approximation
- `plot_policy()` / `plot_value()`: Visualization

---

#### KrusellSmithEnv.py
Implementation of the Krusell-Smith (1998) algorithm for incomplete markets models with aggregate uncertainty. Features:
- Individual-specific and aggregate shocks
- Value function iteration
- Linear approximation of aggregate dynamics
- Monte Carlo simulation

**Key Classes:**
- `KrusellSmithEnv`: Main environment class

**Key Methods:**
- `no_agg_risk_VFI()`: VFI without aggregate uncertainty
- `KS_vfi()`: Full Krusell-Smith algorithm with aggregate uncertainty
- `solve_approximated_equilibrium()`: Find approximated equilibrium
- `simulate_aggk()`: Simulate aggregate capital series
- `forecast_aggk()`: Forecast future aggregate capital

---

#### FuncApproxEnv.py
Environment for comparing different function approximation methods (Problem Set 1, Question 2). Features comparison of:
- Spline approximation
- Polynomial approximation
- Chebyshev approximation

**Key Classes:**
- `FuncApproxEnv`: Main environment class

**Key Methods:**
- `validate()`: Compare approximation methods on test function

---

### Numerical Methods

#### FunctionApprox.py
Various function approximation techniques for numerical economics.

**Key Classes:**
- `FunctionApprox`: Base class
- `ChebyshevApprox`: Chebyshev polynomial approximation (1D)
- `ChebyshevApprox2D`: 2D Chebyshev approximation
- `PolynomialApprox`: Polynomial regression
- `SplineApprox`: Cubic spline interpolation
- `LinearApprox`: Linear approximation
- `ExpLogLinearApprox`: Exponential-log linear approximation

---

#### MarkovApprox.py
Markov chain approximations for AR(1) processes.

**Key Classes:**
- `MarkovApprox`: Base class
- `Rowenhorst`: Rowenhorst method (exact discretization)
- `Tauchen`: Tauchen (1986) method
- `TauchenHussey`: Tauchen-Hussey method

**Key Functions:**
- `markov_moments()`: Compute moments of Markov chain

---

#### RandomProcess.py
Simulation of stochastic processes.

**Key Classes:**
- `MarkovProcess`: Base class
- `AR1Process`: AR(1) process simulation
- `FiniteMarkov`: Finite state Markov chain

---

### Utilities

#### utilize.py
Helper functions for output generation.

**Functions:**
- `write_markdown_table()`: Generate markdown tables
- `write_latex_table()`: Generate LaTeX tables
- `gen_time_series_moments()`: Compute time series moments
- `compute_gini()`: Calculate Gini coefficient

---

#### menu_cost.py
Menu cost model implementation.

**Key Classes:**
- `MenuCostEnv`: Main environment class
- `MenuCostVFIRes`: Results container

**Key Methods:**
- `vfi()`: Value function iteration
- `simulate()`: Simulate price dynamics
- `hazard_function()`: Compute hazard rates

---

## Problem Sets

### Problem Set 1: Optimal Growth & Function Approximation

**Question 1: Solow Growth Model**
- Solves optimal growth model with AR(1) TFP using Chebyshev approximation
- Compares grid search vs. Euler equation methods
- Replicates Table 2 in Kopecky and Suen (2010)
- Key finding: Rowenhorst discretization performs best

**Question 2: Function Approximation Comparison**
- Compares Spline, Polynomial, and Chebyshev methods
- Test function: f(x) = αβx^α with α=0.3, β=0.98
- Chebyshev and Polynomial capture curvature better than Spline

**Question 3: Habit Formation & Investment Adjustment Costs**
- Growth model with internal/external habit formation
- Investment adjustment costs (IAC)
- Analyzes policy functions under different parameter values

---

### Problem Set 2: Stochastic Optimal Growth

Compares four numerical methods for solving stochastic optimal growth:

| Method | σ(y) | σ(c) | σ(i) | 
|:------:|:----:|:----:|:----:|
| VFI | 0.205 | 0.138 | 0.071 |
| Chebyshev | 0.148 | 0.096 | 0.055 |
| Modified PEA | 0.120 | 0.076 | 0.046 |
| EGM | 0.157 | 0.105 | 0.054 |

All methods show high correlation between model variables but underestimate real-world volatility.

---

### Problem Set 3: Huggett (1996) Model

Life-cycle model with incomplete markets:
- 79 periods, retirement at age 54
- Borrowing constraint (a ≥ 0)
- Pension system
- Stationary equilibrium computation

**Key Results:**
- Aggregate capital: K = 3.73
- Wealth Gini: 0.577 (underestimates US inequality)
- Top 1% holds 7.8% of wealth (vs 28% in US data)

**Comparative Statics:**
- Lower population growth → lower output, higher capital
- Welfare effects largest for working-age cohorts

---

### Final Project: Income and Wealth Dynamics

Analyzes income and wealth dynamics using the Krusell-Smith model framework:
- Forecast analysis (1, 10, 50 periods ahead)
- Income and wealth distributions
- Gini coefficients, moments, and inequality measures

**Generated Figures:**
- `forecast_*.png`: Forecasts at different horizons
- `gini_*.png`: Gini coefficients over time
- `policy_*.png`: Policy functions
- `mean_*.png`, `variance_*.png`: Distribution moments
- `skewness_*.png`, `kurtosis_*.png`: Higher-order moments

---

## Installation

### Dependencies
```bash
numpy
scipy
matplotlib
pandas
```

Install via:
```bash
pip install numpy scipy matplotlib pandas
```

## Usage

Run specific problem sets:
```bash
# Problem Set 1
python main.py

# Problem Set 2
python problem_set_2.py

# Problem Set 3
python problem_set_3.py
```

Or import modules directly:
```python
from OptimalGrowth import OptimalGrowthEnv
from Huggett1996Env import Huggett1996Env
from FunctionApprox import ChebyshevApprox

# Example: Solve optimal growth model
env = OptimalGrowthEnv()
env.euler_method(n_k=15, n_a=7)
env.plot_policy()
```

---

## References

1. Kopecky, K.A., & Suen, R.M.H. (2010). Finite State Markov-Chain Approximations to Highly Persistent AR(1) Processes. *Journal of Economic Dynamics and Control*.

2. Huggett, M. (1996). Wealth Distribution in Life-Cycle Economies. *Journal of Monetary Economics*.

3. Krusell, P., & Smith, A.A. (1998). Income and Wealth Heterogeneity in the Macroeconomy. *Journal of Political Economy*.

4. Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*.

5. Rowenhorst, D. (1995). A Exact Procedure for Evaluating Markovian Aggregate Real Business Cycle Models. *Review of Economic Studies*.

