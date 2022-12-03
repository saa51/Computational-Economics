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

The main code is in `OptimalGrowth.py`

#### Solving a optimal growth model with Chebyshev approximation and grid search

#### Solving a optimal growth model with Chebyshev approximation and euler equation

#### Replication of Kopecky and Suen (2010)

| index | T(5) | T(10) | T(25) | TH(5) | TH(10) | TH(25) | R(5) | R(10) | R(25) | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| rho | 0.9467646569644732 | 0.9535801169108573 | 0.9785232888031756 | 0.9584457132537361 | 1.2023962764420566 | 1.3958806599246751 | 0.960249213874119 | 0.9942382865351974 | 1.0 | 
| sigma_epsilon | 0.26152820030007967 | 0.5156300490013266 | 0.7869816046231504 | 0.08797495700626437 | 1.9132339880339997 | 15.95866908322857 | 0.33168844431015765 | 0.5282861671996443 | 1.0 | 
| sigma_a | 0.005178490968907907 | 0.005517902623375869 | 0.01409447310822537 | 2.089779030048299 | 1.5486342715435837 | 1.3139241178069108 | 1.0000146110564414 | 0.9999951529493425 | 1.0 | 
| sigma_k | 1.2696974318647773 | 1.3346697384259643 | 1.0285151407977213 | 0.8907612977919973 | 0.9881521633394109 | 0.8123022943625443 | 1.000006444124794 | 1.0000010610131775 | 1.0 | 
| sigma_ak | 1.234364228059926 | 1.2930308028204118 | 1.0280232306254495 | 0.8907660093110623 | 0.23788344175521564 | 0.614591504352474 | 1.000006374947611 | 1.000000756899851 | 1.0 | 
| sigma_y | 1.0833787608555057 | 1.1041322173888763 | 1.0094212725835332 | 0.9635737019371478 | 0.8190400820101913 | 0.8911791370956246 | 1.0000021371198269 | 1.000000289198325 | 1.0 | 
| sigma_c | 1.0788126165121796 | 1.0986451610622858 | 1.0087848758909013 | 0.9662690324824407 | 0.8528979216304867 | 0.9061652518055084 | 1.0000019764526333 | 1.000000216635173 | 1.0 | 
| sigma_i | 1.2736068168740995 | 1.3389684348034732 | 1.024334436503337 | 0.8902780891229237 | 1.976345475306701 | 1.3648340085961401 | 1.0000092866327386 | 1.0000006944889794 | 1.0 | 
| rho_y | 0.9999997329131763 | 0.9999996558431127 | 0.9999999816175059 | 1.000000055662889 | 0.9999983752810665 | 0.9999994677665356 | 0.9999999999980074 | 0.9999999999995393 | 1.0 | 

### Question 2

The main code is in `FuncApproxEnv.py`.

`FuncApproxEnv().validate(n)`: Compare Spline, Polynomial, and Cherbyshev approximation with `n` grids.

### Question 3