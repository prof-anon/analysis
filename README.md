# Latency Penalty and Relationship with Inclusion Likelihood
1. The `latency` folder contains the script and data to reproduce results from Section 6.2 ``Effect of Latency on Inclusion''
2. Run as follows:
```
cd latency
python latency.py
```
3. It will take 3-5 min to finish generating all the plots.

# Simulation based Economic Analysis of various mechanisms
1. The `sim` folder contains the script to reproduce results from Section 5.1 ``Economic Utility for PROF Users''

2. Run as follows:
```
cd sim
python examples.py
```
3. It takes around 1 day to finish.

# Demand Ratios of AMM data
1. The `data` folder contains the script to reproduce results from Appendix F ``Validation of Our Model using Real World Data''
2. Run as follows:
```
cd data
python data.py
``` 
3. Downloads swap data from Uniswapv3 and Sushiswap
4. Calculates the demand ratio for them
5. Plots the ratio for the top Uniswapv3 and Sushiswap pools