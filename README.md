# Computational-Economics

## Code List

`main.py`: where the code should be run.

`OptimalGrowth.py`: Provide a optimal growth model environment and solutions, corresponding to Question 1 in Problem set 1.

`FuncApproxEnv.py`: Provide an environment in which Spline, Polynomial, and Cherbyshev approximation are compared, corresponding to Question 2 in Problem set 1.

`GrowthWithHabitIACEnv.py`: Provide a growth model with habit utility and investment adjustment cost, corresponding to Question 3 in Problem set 1.

`FunctionApprox.py`: Some function approximation tools.

`MarkovApprox.py`: Using finite state Markov process to approximate AR(1) process.

`RandomProcess.py`: Simulation of some random process, AR(1) only up to now.

`utilize.py`: some tools.

## Problem Set 1

### Quetion 1

This is a Solow growth model with AR(1) TFP in Kopechy and Suen (2010).
I use the parameters in their paper.
The main code is in `OptimalGrowthEnv.py`.

#### Solving a optimal growth model with Chebyshev approximation and grid search

The main code for this problem is `OptimalGrowthEnv().grid_search()`.
I use Chebyshev method to approximate the value function, and search for the optimal consumption to maximize the RHS of the Bellman equation.

The AR(1) TFP is approximated by a 7-state Markov process using Rowenhorst method.
The number of grids for capital is 15.
The domain of the solution is [0.5, 1.5] * steady state capital.
And the precision of optimal consumption is 0.01.

Following is the optimal consumption, capital accumulation, and value function: 

![image](figures/problem_set_1/q12_v.png)
![image](figures/problem_set_1/q12_c.png)
![image](figures/problem_set_1/q12_k.png)
The X-axis is for capital, and the lines with different colors represent different TFP levels.

#### Solving a optimal growth model with Chebyshev approximation and euler equation

The main code for this problem is in `OptimalGrowthEnv().Euler_method()`.
I use Chebshev method to approximate the consumption function given present capital and TFP.
Then I use RHS of Euler equation to update the consumption function.

The setting and parameters are the same as the last problem.
Following is the solution:

![image](figures/problem_set_1/q13_v.png)
![image](figures/problem_set_1/q13_c.png)
![image](figures/problem_set_1/q13_k.png)

The red points are the evaluation on the grid points.

Since the grid width is sufficiently small, it seems that there is not significant difference between the solution 2 methods.
But the consumption seems to be less smooth in grid search.
And the first method is more time-consuming.

#### Replication of Table 2 in Kopecky and Suen (2010)

I use the euler method to replicate Table 2 in their paper.
I compute the first 3 lines by the transition probability matrix.
The remaining statistics are computed by Monte-Carlo simulation.
I choose Rowenhorst with 25 grids to be the quasi-exact solution rather than the ChebshevPEA solution in their paper.
The simulation length is 5,010,000, and the first 10,000 periods are burned.
Following is the main results.

| index | T(5) | T(10) | T(25) | TH(5) | TH(10) | TH(25) | R(5) | R(10) | R(25) | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| rho | 0.9489813005320843 | 0.9558585784302387 | 0.9772729323983873 | 0.9584457132537361 | 1.2023962764420566 | 1.3958806599246751 | 0.960249213874119 | 0.9942382865351974 | 1.0 |
| sigma_epsilon | 0.7792337727919413 | 1.0862579845810687 | 1.010657101453192 | 0.0006353786225143621 | 1.8439167794230527 | 2.817079034961109 | 0.9595960213868383 | 0.9918961180434522 | 1.0 |
| sigma_a | 1.0000473536049053 | 0.9999960470162034 | 0.9999945511175873 | 2.089779030048299 | 1.5486342715435837 | 1.3139241178069108 | 1.0000146110564414 | 0.9999951529493425 | 1.0 |
| sigma_k | 0.8980602752028246 | 1.046804736558316 | 1.0094278111060933 | 0.640532387594404 | 0.5516469891459609 | 0.5551399026945608 | 1.0000194935978146 | 1.0000061654351353 | 1.0 | 
| sigma_ak | 0.8979272520389272 | 1.0468215152083467 | 1.0094404187222084 | 0.6404409052628426 | 0.20032107276818503 | 0.4742085794630114 | 1.0000210551390718 | 1.000006632563311 | 1.0 | 
| sigma_y | 0.9643723894330363 | 1.016457681510672 | 1.0033115758511089 | 0.8761224950010456 | 0.7601242660502582 | 0.8284682390792939 | 1.000007147128859 | 1.0000022552190273 | 1.0 | 
| sigma_c | 0.9894466705978897 | 1.0056940913812102 | 1.0011155230106046 | 0.9696329346187078 | 1.0547732697227472 | 0.9972331678651093 | 1.0000025311878658 | 1.0000007945277887 | 1.0 | 
| sigma_i | 0.8969417599998663 | 1.0465165503950191 | 1.0093518616179082 | 0.6390693726384001 | 0.8788448685123512 | 0.6964464758441952 | 1.0000218465266735 | 1.0000068212979027 | 1.0 | 
| rho_y | 0.9999999955611639 | 1.000000001799816 | 1.0000000003912346 | 0.9999999713844252 | 0.9999997828497693 | 0.9999999143962336 | 1.0000000000004567 | 1.0000000000000082 | 1.0 | 

The results are similar to Kopecky and Suen's paper.
In the case of moment approximation, Rowenhorst's performance is best and Tauchen Hussey is the worst.
As the number of grids increases, Tauchen and Rowenhorst converges to the quasi-exact results.
The Tauchen Hussay method doesn't converge as the original paper shows.
The reason may be that Python suffers more from numerical errors than Matlab.
### Question 2

The main code is in `FuncApproxEnv.py`.

`FuncApproxEnv().validate(n)`: Compare Spline, Polynomial, and Cherbyshev approximation with `n` grids.
The approximated function is f(x) = alpha * beta * x ^ alpha, with alpha=0.3 and beta=0.98.

The approximation result is following:
![image](figures/problem_set_1/q2_5.png)
![image](figures/problem_set_1/q2_10.png)

The computing time and maximum absolute error is in the following table:

| Method | Spline(5) | Polynomial(5) | Chebyshev(5) | Spline(10) | Polynomial(10) | Chebyshev(10) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Computational Time | 0.0 | 0.0009911060333251953 | 0.0 | 0.0 | 0.0 | 0.0010030269622802734 |
| Maximum Absolute Error | 0.16146569818991097 | 0.12879991407126204 | 0.1579644117293864 | 0.12678461317265927 | 0.09112689484861211 | 0.10377239873151732|

The computional time is neglectable since the number of basis function is small.

Chebyshev and Polynomial approximation can capture the curvature of the function, so both of them perform better than Spline in term of errors.
This function is kind of ill-behaved because that its all order of derivatives is infinite at 0, so the information at 0 is important.
Because Chebyshev approximation doesn't take grids at the endpoints, so there is a large error at 0. 
### Question 3