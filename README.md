# Prof Analysis

# Prereq

* python3 must be installed
* python3 libraries `pip3 install -r requirements.txt`

# Latency Penalty and Relationship with Inclusion Likelihood
1. The `latency` folder contains the script and data to reproduce results from Section 6.2 ``Effect of Latency on Inclusion''
2. Run as follows:
```
cd latency
python latency.py
```
3. It will take 3-5 min to finish generating all the plots.

# Simulation based Economic Analysis of various mechanisms
The `sim` folder contains the script to reproduce results from Section 5.1 ``Economic Utility for PROF Users''

## Get simultion data from public link and generate graphs.
```
cd sim
wget -O sim_data.zip https://osf.io/download/93n4s/?view_only=5cd5a58b51054388a9025acb5c59caac 
unzip sim_data.zip -d data_archive
python3 examples.py data_archive/paper
```

## Regenerate the simulation data used in the paper and generate graphs. It takes around 1 day to finish generating the simulation data.
```
cd sim
mkdir data_gen
python3 examples.py data_gen
```

# Demand Ratios of AMM data
The `data` folder contains the script to reproduce results from Appendix F ``Validation of Our Model using Real World Data''

## Generate graphs for figure based on demand ration data 

```
cd data
python3 data.py demand_ratio_paper
``` 

## Regenerate demand ration data
To re-download real world AMM data from the Ethereum blockchain you need an [Infura Account](https://support.infura.io/account/api-keys/create-new-key) you must set `INFURA_SECRET` and `INFURA_API_KEY` environment variables based on your account credentials

```
export INFURA_SECRET=<INFURA_SECRET>
export INFURA_API_KEY=<INFURA_API_KEY>
cd data
mkdir data_gen
python3 data.py data_gen
``` 

This script will:

1. download a list of top 2 pages of tokens on [Etherscan](https://etherscan.io/tokens)
2. download AMM pool addresses Uniswapv3 and Sushiswap and filter for only pools that swap the top tokens
3. downloads swap data for these AMM pools
4. calculates the demand ratio for them
5. plots the ratio for the top Uniswapv3 and Sushiswap pools