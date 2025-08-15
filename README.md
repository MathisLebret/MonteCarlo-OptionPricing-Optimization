# From Naive Python to Optimized C++: Monte Carlo Option Pricing

## Project Overview

In algorithmic finance, fast reaction to market information is crucial. This project investigates the impact of implementation choices on European call option pricing using Monte Carlo simulations. We compare naive Python implementations with optimized C++ code, leveraging OpenMP for parallelization and EigenRand for vectorized random number generation. Performance tests show that the optimized C++ version achieves a speedup of approximately 3,000× compared to naive Python and 40× compared to a NumPy-based approach.

While Monte Carlo simulation is excessive for such simple payoffs, the study remains relevant for path-dependent derivatives. The project also includes an analysis of the convergence of the pricing estimator, highlighting the practical trade-offs between accuracy and computational efficiency.

## Files included

1. A **LaTeX report** ('Project_report.pdf') containing both the mathematical background and application results.
2. A **Jupyter notebook** ('Python_simulations.ipynb') with the simulations in Python and a plot for the estimator convergence.
3. A **C++ file** ('CPP_simulations.cpp') including all the simulations in C++ and the code used for convergence analysis.
   
## Libraries used

- **Python:** NumPy (vectorization), ScipyStats (random number generation), Matplotlib. 
- **C++:** Eigen Dense (linear algebra), EigenRand (vectorized random number generation), OpenMP (parallelization).

The external libraries were used to optimize performance; credits are available at the end of the LaTeX report.

## What I learned through this project

This personal project allowed me to carry out a hands-on practical study covering:
- **Quantitative Finance**: Black–Scholes model, option pricing  
- **Numerical Methods**: Euler–Maruyama scheme, Monte Carlo simulation, estimator convergence  
- **Programming**: Python, C++, optimization  

I discovered key libraries used by Quantitative Developers (NumPy, Eigen, EigenRand) and strengthened my understanding of statistical models.

## Author

**Mathis Lebret** - *Second Year Student in a Joint MSc in Quantitative Finance & Risk Engineering at ISFA Lyon and Ecole Centrale de Lyon.*\
Link to my **[LinkedIn Profile](https://www.linkedin.com/in/mathis-lebret-566952190)**.