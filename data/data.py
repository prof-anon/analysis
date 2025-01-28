import math
import json
import requests
import os
import time
import logging
import sys 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eth_abi import abi
from bs4 import BeautifulSoup

BLOCK_NUMBER_MIN=15537572
BLOCK_NUMBER_MAX=18473541
BLOCK_DATA_FOLDER = "swap_data"
CONTRACT_DATA_FOLDER = "contract_data"
DEMAND_RATIO_DATA_FOLDER = "demand_ratio"
POOL_DATA_FOLDER = "pool_data"
TOKEN_DATA_FOLDER = "token_data"
MEV_DATA_FOLDER = "mev_data"
FIGURE_FOLDER = "figure"
LOG_FOLDER = "log"
INFURA_API_KEY = os.environ.get("INFURA_API_KEY")
INFURA_SECRET = os.environ.get("INFURA_SECRET")
INFURA_URL = f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"
NONATOMIC_DATASET="1b3TafqpVzP88VhK9UT8oj49jsHee2yXt"

UNISWAPV3_TOPIC = {
    "create_pool": "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118",
    "create_pool_data_types":['int24', 'address'],
    "create_pool_topic_types": [None, ['address'], ['address'], ['uint24']],
    "pair_name_foo": lambda topics, token_names: f"{token_names(topics[1])}_{token_names(topics[2])}_{topics[3]/10000}",
    "pool_address_foo": lambda create_event: create_event[1],
    "swap": "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",
    "swap_data_types": ['int256', 'int256', 'uint160', 'uint128', 'int24'],
    "swap_data_names": ['amount0','amount1','price','liq','tick'],
}

UNISWAPV2_TOPIC = {
    "create_pool": "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9",
    "create_pool_data_types":['address', "uint256"],
    "create_pool_topic_types": [None, ['address'], ['address']],
    "pair_name_foo": lambda topics, token_names: f"{token_names[topics[1].lower()]}_{token_names[topics[2].lower()]}",
    "pool_address_foo": lambda create_event: create_event[0],
    "swap": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
    "swap_data_types": None, #TODO
    "swap_data_names": None, #TODO
}

POOL_FACTORY = {
    "sushiswap": {
        "factory": "0xbACEB8eC6b9355Dfc0269C18bac9d6E2Bdc29C4F", 
        "topic": UNISWAPV3_TOPIC
    },
    "uniswapv3": {
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984", 
        "topic": UNISWAPV3_TOPIC
    },
    # "uniswapv2": {
    #     "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f", 
    #     "topic": UNISWAPV2_TOPIC
    # }
}

FILTERED_TAG = {True: "filtered", False: "unfiltered", None: "all"}
FIGURE_NUM = 0


HISTOGRAM_BINS = {
    "0-8+" : { 
        "not enough samples\nin block": lambda x: x==-1, 
        "0": lambda x: x==0, 
        "(0-0.25]": lambda x: x > 0 and x <= .25,
        "(0.25-0.50]": lambda x: x > .25 and x <= .5,
        "(0.50-0.75]": lambda x: x > .5 and x <= .75,
        "(0.75-1.0]": lambda x: x > .75 and x <= 1,
        "(1.0-2.0]": lambda x: x > 1 and x <= 2,
        "(2.0-4.0]": lambda x: x > 2 and x <= 4,
        "(4.0-8.0]": lambda x: x > 4 and x <= 8,
        "(8.0+)": lambda x: x > 8,
        "std=0": lambda x: x==-2,
    },
    "0-2+": { 
        "not enough samples\nin block": lambda x: x==-1, 
        "0": lambda x: x==0, 
        "(0-0.25]": lambda x: x > 0 and x <= .25,
        "(0.25-0.50]": lambda x: x > .25 and x <= .5,
        "(0.50-0.75]": lambda x: x > .5 and x <= .75,
        "(0.75-1.0]": lambda x: x > .75 and x <= 1,
        "(1.0-1.25]": lambda x: x > 1 and x <= 1.25,
        "(1.25-1.5]": lambda x: x > 1.25 and x <= 1.5,
        "(1.5-1.75]": lambda x: x > 1.5 and x <= 1.75,
        "(1.75-2]": lambda x: x > 1.75 and x <= 2,
        "(2.0+)": lambda x: x > 2,
        "std=0": lambda x: x==-2,
    }
}

def get_top_tokens(TOKEN_DATA_FOLDER=TOKEN_DATA_FOLDER):
    if not os.path.isdir(TOKEN_DATA_FOLDER):
        os.makedirs(TOKEN_DATA_FOLDER)
    num_pages = 2
    tokens_info = {"0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH"}
    for i in range(num_pages):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15'}
        res = requests.get(f"https://etherscan.io/tokens?p={i+1}", headers=headers)
        if res.status_code != 200:
            exit(res.status_code)
        data = res.text 
        soup = BeautifulSoup(data, 'html.parser')
        with open(f"{TOKEN_DATA_FOLDER}/page{i+1}.html", "w") as f:
            f.write(soup.prettify())
        tokens = soup.find_all(id='ContentPlaceHolder1_tblErc20Tokens')[0].find_all('a')
        for tag in tokens:
            token_address = tag.get("href")
            if '/token/0x' in token_address:
                token_symbol = tag.find_all(class_="text-muted")[0].contents[0][1:-1]
                address = token_address.replace('/token/', '').lower()
                tokens_info[address] = token_symbol
                print(address, token_symbol)
    json.dump(tokens_info,open(f"{TOKEN_DATA_FOLDER}/top_tokens.json", "w"), indent=2)

def retry_with_block_range(make_request_foo, check_request_foo, parse_result_foo, data_dir, block_start, block_end, block_range=800):
    if not os.path.isdir(LOG_FOLDER):
        os.mkdir(LOG_FOLDER)
    if "/" in data_dir:
        tag = data_dir[:data_dir.find('/')]
    else:
        tag = data_dir
        data_dir += "/"
    logging.basicConfig(filename=f"{LOG_FOLDER}/{tag}.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
    
    progress_file = f"{LOG_FOLDER}/progress_{tag}"
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            block_start=int(f.read())+1
        logging.warning(f"actually starting at {block_start} to {block_start + block_range}")
        logging.warning(f"delete {progress_file} to start downloading from scratch")
    if block_start > block_end:
        return

    orig_block_range=block_range
    num_range_large = 0
    num_range_ok = 0
    max_times_range = 5
    
    curr_time = time.time()
    block_curr = block_start
    block_next = block_start + block_range

    while block_curr == "earliest" or block_curr < block_end:
        res = make_request_foo(block_curr, block_next)
        success = check_request_foo(res)
        duration = time.time() - curr_time
        if success:
            df = parse_result_foo(res)
            filename=f'{data_dir}{block_curr}_{block_next}.pkl'
            df.to_pickle(filename)
            
            logging.info("%.2fs %s done, blocks left %d queries left: %d", [round(duration,2), filename,block_end-block_next,(block_end-block_next)/block_range])
            with open(progress_file, "w") as f:
                f.write(str(block_next))
            block_curr=block_next+1
            block_next+=block_range

            num_range_large=0
            num_range_ok+=1
            if num_range_ok>=max_times_range and block_range < orig_block_range:
                old_block_range = block_range
                block_range = max(orig_block_range, int(block_range*2))
                logging.info(f"adjusting range from {old_block_range} to {block_range}")
        else:
            num_range_ok=0
            num_range_large+=1
            old_range = f"{block_curr}-{block_next}"
            block_next = block_curr + int((block_next - block_curr)/2)
            logging.warning(f"retry with [{old_range}] -> [{block_curr}-{block_next}] length {block_next-block_curr}")
            if num_range_large>=max_times_range:
                old_block_range = block_range
                block_range = int(block_range/2)
                assert block_range > 0
                logging.warning(f"adjusting range from {old_block_range} to {block_range}")
        curr_time = time.time()
    logging.info(f"delete {progress_file} to start downloading from scratch")

def make_request_infura_foo(url, jsonpayload, headers, auth):
    assert len(jsonpayload['params']) == 1

    def make_request_infura(block_start, block_end):
        if block_start != "earliest":
            block_start = hex(block_start)
        if block_end != "latest":
            block_end = hex(block_end)
        jsonpayload['params'][0]['fromBlock'] = block_start
        jsonpayload['params'][0]['toBlock'] = block_end
        res = requests.post(url=url, json=jsonpayload, headers=headers, auth=auth)
        return res
        
    return make_request_infura

def check_request_infura(res):
    if res.status_code == 200:
        data = res.json()
        if 'error' in data:
            return False
        else:
            return True
    else:
        print(res.status_code, res.text)
        exit(res.status_code)

def parse_request_infura(res):
    data = res.json()
    df = pd.DataFrame.from_records(data["result"])
    df["blockNumber"] = df['blockNumber'].apply(int, base=16)
    df["logIndex"] = df['logIndex'].apply(int, base=16)
    df["transactionIndex"] = df['transactionIndex'].apply(int, base=16)
    return df

def download_create_pool_topic(POOL_DATA_FOLDER=POOL_DATA_FOLDER):
    assert INFURA_API_KEY != "", "set INFURA_API_KEY environment variable"
    assert INFURA_SECRET != "", "set INFURA_SECRET environment variable"
    for protocol in POOL_FACTORY:
        if not os.path.isdir(f"{POOL_DATA_FOLDER}"):
            os.makedirs(f"{POOL_DATA_FOLDER}")
        factory_address = POOL_FACTORY[protocol]["factory"]
        topic = POOL_FACTORY[protocol]["topic"]["create_pool"]
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "id":1,
            "params":[{
                "fromBlock": hex(BLOCK_NUMBER_MIN),
                "toBlock": hex(BLOCK_NUMBER_MAX), 
                "address": factory_address, 
                "topics": [topic]
                }
            ]
        }
        headers = {}
        auth = (INFURA_API_KEY, INFURA_SECRET)
        make_request_foo = make_request_infura_foo(INFURA_URL, payload, headers, auth)
        retry_with_block_range(make_request_foo, check_request_infura, parse_request_infura, f"{POOL_DATA_FOLDER}/create_pool_topic_{protocol}_", BLOCK_NUMBER_MIN, BLOCK_NUMBER_MAX, block_range=1000)

def parse_create_pool_topic(POOL_DATA_FOLDER=POOL_DATA_FOLDER, TOKEN_DATA_FOLDER=TOKEN_DATA_FOLDER):
    top_tokens = json.load(open(f"{TOKEN_DATA_FOLDER}/top_tokens.json"))

    def address_to_token(token):
        token=token.lower()
        if token in top_tokens:
            return top_tokens[token]
        return token
    pool_files = os.listdir(POOL_DATA_FOLDER)
    num_pools = len(pool_files)
    top_pools = {}
    for protocol in POOL_FACTORY:
        top_pools[protocol] = {}
    for pi in range(len(pool_files)):
        pool_data_filename = pool_files[pi]
        s1 = pool_data_filename.find('create_pool_topic_') + len('create_pool_topic_')
        e1 = pool_data_filename[s1:].find('_')+s1
        protocol = pool_data_filename[s1:e1]
        if protocol not in POOL_FACTORY:
            continue
        print(f"parse_create_pool_topic: {pi+1}/{num_pools} pool_data_filename", pool_data_filename)
        dataTypesArray = POOL_FACTORY[protocol]['topic']['create_pool_data_types']
        topicTypesArray = POOL_FACTORY[protocol]['topic']['create_pool_topic_types']
        factorydata = pd.read_pickle(f"{POOL_DATA_FOLDER}/{pool_data_filename}")
        def decodePoolAddress(data):
            params = abi.decode(dataTypesArray, bytes.fromhex(data[2:]))
            return POOL_FACTORY[protocol]['topic']['pool_address_foo'](params)
        def decodePairName(topics):
            all_topics = []
            for i in range(len(topics)):
                t = topics[i]
                if topicTypesArray[i] is None:
                    all_topics.append(None)
                else:
                    all_topics += abi.decode(topicTypesArray[i], bytes.fromhex(t[2:]))
            return POOL_FACTORY[protocol]['topic']['pair_name_foo'](all_topics, address_to_token)

        factorydata["poolAddress"] = factorydata["data"].apply(decodePoolAddress)
        factorydata["pairName"] = factorydata["topics"].apply(decodePairName)
        pools = factorydata[["poolAddress", "pairName"]].set_index("poolAddress")
        result = pools["pairName"].to_dict()
        top_pools[protocol] = {**top_pools[protocol], **result}
    json.dump(result, open(f"{POOL_DATA_FOLDER}/top_pools.json", "w"), indent=2)
    
def download_swap_topic(BLOCK_DATA_FOLDER=BLOCK_DATA_FOLDER):
    assert INFURA_API_KEY != ""
    assert INFURA_SECRET != ""
    if not os.path.isdir(f"{BLOCK_DATA_FOLDER}v3/"):
        os.makedirs(f"{BLOCK_DATA_FOLDER}v3/")
    # if not os.path.isdir(f"{BLOCK_DATA_FOLDER}v2/"):
    #     os.makedirs(f"{BLOCK_DATA_FOLDER}v2/")

    block_end= BLOCK_NUMBER_MAX
    block_start = BLOCK_NUMBER_MIN
    headers = {}
    auth = (INFURA_API_KEY, INFURA_SECRET)
    payloadv3 = {
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "id" : 1,
        "params": [{
            "fromBlock": hex(block_start), 
            "toBlock": hex(block_end), 
            "topics": [UNISWAPV3_TOPIC["swap"]]
        }]
    }
    make_request_foov3 = make_request_infura_foo(INFURA_URL, payloadv3, headers, auth)
    retry_with_block_range(make_request_foov3, check_request_infura, parse_request_infura, f"{BLOCK_DATA_FOLDER}v3/swap_topicv3_", block_start, block_end, block_range=800)

    # payloadv2 = {
    #     "jsonrpc": "2.0",
    #     "method": "eth_getLogs",
    #     "id" : 1,
    #     "params": [{
    #         "fromBlock": hex(block_start), 
    #         "toBlock": hex(block_end), 
    #         "topics": [UNISWAPV2_TOPIC["swap"]]
    #     }]
    # }
    # make_request_foov2 = make_request_infura_foo(INFURA_URL, payloadv2, headers, auth)
    # retry_with_block_range(make_request_foov2, check_request_infura, parse_request_infura, f"{BLOCK_DATA_FOLDER}v2/swap_topicv2", block_start, block_end, block_range=800)

def combine_chunks(combine_df_foo, files_dir, files_list, tmp_dir, output_dir, size_chunk):
    if not os.path.isdir(LOG_FOLDER):
        os.mkdir(LOG_FOLDER)
    progress_file = f"{LOG_FOLDER}/progress_{files_dir}_combine_chunks"
    iter_num = 0
    
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            last_iter=int(f.read())
        files_dir = f"{tmp_dir}/tmp_chunks{last_iter}"
        files_list = os.listdir(files_dir)
        iter_num = last_iter+1
        print(f"actually starting at {iter_num}")
        print(f"delete {progress_file} to start combining from scratch")    
    
    num_files = len(files_list)
    while num_files >= size_chunk:
        chunk_folder = f"{tmp_dir}/tmp_chunks{iter_num}"
        print(f"{iter_num}: combining {files_dir} into {chunk_folder}")
        if not os.path.isdir(chunk_folder):
            os.makedirs(chunk_folder)

        for i in range(0, num_files-size_chunk, size_chunk):
            combine_df_foo([f"{files_dir}/{f}" for f in files_list[i:i+size_chunk]], f"{int(i/size_chunk)}_", chunk_folder)
        if iter_num > 0:
            print(f"{iter_num}: removing {files_dir}")

            for f in files_list:
                os.remove(f"{files_dir}/{f}")
            os.rmdir(files_dir)

        with open(progress_file, "w") as f:
            f.write(str(iter_num))

        iter_num+=1
        files_list = os.listdir(chunk_folder)
        files_dir = chunk_folder
        num_files = len(files_list)

    combine_df_foo([f"{files_dir}/{f}" for f in files_list], "", output_dir)
    
def download_mev_dataset():
    if not os.path.isdir(MEV_DATA_FOLDER):
        os.makedirs(MEV_DATA_FOLDER)
    if not os.path.isdir(f"{MEV_DATA_FOLDER}_zeromev"):
        os.makedirs(f"{MEV_DATA_FOLDER}_zeromev")

    column_names = [
        "block_number", 
        "tx_index",
        "mev_type",
        "protocol",
        "user_loss_usd",
        "extractor_profit_usd",
        "user_swap_volume_usd",
        "user_swap_count",
        "extractor_swap_volume_usd",
        "extractor_swap_count",
        "imbalance",
        "address_from",
        "address_to",
        "arrival_time_us",
        "arrival_time_eu",
        "arrival_time_as"
    ]
    def make_request(block_start, block_end):
        params = {"block_number": block_start, 'count': block_end-block_start}
        headers = {'Accept':'application/json'}
        res = requests.get("https://data.zeromev.org/v1/mevBlock", params=params, headers=headers, timeout=10)
        time.sleep(1)
        return res
    
    def check_request(res):
        if res.status_code == 200:
            return True
        else:
            print("error", res.status_code, res.text)
            exit(res.status_code)
    
    def parse_request(res):
        df = pd.DataFrame(res.json(), columns=column_names)
        df = df[df['mev_type'].isin(['arb', 'frontrun', 'backrun'])]
        df = df.rename(columns={"block_number": "blockNumber", "tx_index": "transactionIndex"})
        return df
    
    retry_with_block_range(make_request, check_request, parse_request, f"{MEV_DATA_FOLDER}_zeromev/zeromev_", BLOCK_NUMBER_MIN, BLOCK_NUMBER_MAX, block_range=100)

    def combine_df(files_list, chunk_name, output_folder):
        dfs = []
        for f in files_list:
            file_df = pd.read_pickle(f)
            dfs.append(file_df)
        df = pd.concat(dfs, axis=0 , ignore_index=True, sort=False)
        df.to_pickle(f"{output_folder}/{chunk_name}zeromev.pkl")

    # combine_chunks(combine_df, f"{MEV_DATA_FOLDER}_zeromev", os.listdir(f"{MEV_DATA_FOLDER}_zeromev"), f"{MEV_DATA_FOLDER}_zeromev_tmp", MEV_DATA_FOLDER, 1000)
    combine_df([f"{MEV_DATA_FOLDER}_zeromev/{f}" for f in os.listdir(f"{MEV_DATA_FOLDER}_zeromev")], "", MEV_DATA_FOLDER)

    with open(f"cleanup_files.sh", "a") as f:
        f.write(f"rm -rf {MEV_DATA_FOLDER}_zeromev_tmp\n")
    print("run $bash cleanup_files.sh")

def parse_swap_topic(CONTRACT_DATA_FOLDER=CONTRACT_DATA_FOLDER):
    nonAtomicSwaps = pd.read_pickle('filteredSwapsWithPrice.pkl')
    mevtxs = pd.read_pickle(f"{MEV_DATA_FOLDER}/zeromev.pkl")
    mevtxs['mev_tx_tag'] = mevtxs['blockNumber'].astype(str) + "_" + mevtxs['transactionIndex'].astype(str)
    filter_block_min = min(min(nonAtomicSwaps['block']), min(mevtxs['blockNumber']))
    filter_block_max = max(max(nonAtomicSwaps['block']), max(mevtxs['blockNumber']))
    top_pools = json.load(open(f"{POOL_DATA_FOLDER}/top_pools.json"))
    include_unf = (filter_block_min != BLOCK_NUMBER_MIN) or (filter_block_max != BLOCK_NUMBER_MAX) 
    for t in FILTERED_TAG.values():
        if not include_unf and t == FILTERED_TAG[False]:
            continue
        if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{t}_parse"):
            os.makedirs(f"{CONTRACT_DATA_FOLDER}_{t}_parse")

    progress_file = f"{LOG_FOLDER}/parse_swap_topic_progress"
    start = 0
    if os.path.isfile(progress_file):
        with open(progress_file) as f:
            start = int(f.read())+1

    block_files = os.listdir(f"{BLOCK_DATA_FOLDER}v3")
    num_block_files = len(block_files)
    typesArray = UNISWAPV3_TOPIC["swap_data_types"]
    column_names = UNISWAPV3_TOPIC["swap_data_names"]
    for fi in range(start, num_block_files):
        block_data_filename = block_files[fi]
        print(f"parse_swap_topicv3 {fi+1}/{num_block_files} parsing block_data_filename: {block_data_filename}")
        
        dfBlock = pd.read_pickle(f"{BLOCK_DATA_FOLDER}v3/{block_data_filename}")
        decodedData = dfBlock['data'].apply(lambda x: list(abi.decode(typesArray, bytes.fromhex(x[2:]))))
        decodedDataDF = pd.DataFrame(decodedData.tolist(), columns=column_names)

        dfBlock["poolName"] = dfBlock['address'].apply(lambda x: top_pools[x] if x in top_pools else x)
        dfBlockFinal = pd.concat([dfBlock, decodedDataDF], axis=1)
        dfBlockFinal['mev_tx_tag'] = dfBlockFinal['blockNumber'].astype(str) + "_" + dfBlockFinal['transactionIndex'].astype(str)

        dfBlockFiltered = dfBlockFinal[
            (dfBlockFinal['blockNumber'] >= filter_block_min) & 
            (dfBlockFinal['blockNumber'] <= filter_block_max) & 
            (dfBlockFinal['transactionHash'].isin(nonAtomicSwaps['hash']) == False) &
            ( dfBlockFinal['mev_tx_tag'].isin(mevtxs['mev_tx_tag']) == False)]

        for pool_name in dfBlockFinal["poolName"].unique():
            dfAllContract = dfBlockFinal[dfBlockFinal['poolName'] == pool_name]
            if len(dfAllContract) > 0:
                if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[None]}_parse/{pool_name}"):
                    os.makedirs(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[None]}_parse/{pool_name}")
                print("parse_blocks_swaps",f"{pool_name}_{FILTERED_TAG[None]}_{fi}.pkl", len(dfAllContract))
                dfAllContract.to_pickle(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[None]}_parse/{pool_name}/{pool_name}_{FILTERED_TAG[None]}_{fi}.pkl")

        for pool_name in dfBlockFiltered["poolName"].unique():
            dfFilteredContract = dfBlockFiltered[dfBlockFiltered['poolName'] == pool_name]
            if len(dfFilteredContract) > 0:
                if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[True]}_parse/{pool_name}"):
                    os.makedirs(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[True]}_parse/{pool_name}")
                print("parse_blocks_swaps",f"{pool_name}_{FILTERED_TAG[True]}_{fi}.pkl", len(dfFilteredContract))
                dfFilteredContract.to_pickle(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[True]}_parse/{pool_name}/{pool_name}_{FILTERED_TAG[True]}_{fi}.pkl")

        if include_unf:
            dfBlockUnf = dfBlockFinal[
                (dfBlockFinal['blockNumber'] >= filter_block_min) & 
                (dfBlockFinal['blockNumber'] <= filter_block_max)]
            for pool_name in dfBlockUnf["poolName"].unique():
                dfUnfilteredContract = dfBlockUnf[dfBlockUnf['poolName'] == pool_name]
                if len(dfUnfilteredContract) > 0:
                    if not os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[False]}_parse/{pool_name}"):
                        os.makedirs(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[False]}_parse/{pool_name}")
                    print("parse_blocks_swaps",f"{pool_name}_{FILTERED_TAG[False]}_{fi}.pkl", len(dfUnfilteredContract))
                    dfUnfilteredContract.to_pickle(f"{CONTRACT_DATA_FOLDER}_{FILTERED_TAG[False]}_parse/{pool_name}/{pool_name}_{FILTERED_TAG[False]}_{fi}.pkl")
        with open(progress_file, "w") as f:
            f.write(str(fi))


def combine_contract_swaps(CONTRACT_DATA_FOLDER=CONTRACT_DATA_FOLDER):
    if not os.path.isdir(LOG_FOLDER):
        os.mkdir(LOG_FOLDER)
    for filter_tag in FILTERED_TAG.values():
        output_dir = f"{CONTRACT_DATA_FOLDER}_{filter_tag}"
        data_dir = f"{CONTRACT_DATA_FOLDER}_{filter_tag}_parse"
        progress_file = f"{LOG_FOLDER}/combine_contract_swaps_{filter_tag}_progress"
        if not os.path.isdir(data_dir):
            print(f"ignoring... {data_dir}_parse")
            continue
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        all_pool_names = os.listdir(data_dir)
        num_pools = len(all_pool_names)
        start_pi = 0
        if os.path.isfile(progress_file):
            with open(progress_file) as f:
                start_pi = int(f.read())
            
        for pi in range(start_pi, num_pools):
            pool_name = all_pool_names[pi]
            print(f"combine_contract_swaps {pi+1}/{num_pools} combining pool_name {pool_name} {filter_tag}")

            def combine_df(file_list, chunk_name, output_folder):
                df_list= []
                for f in file_list:
                    df = pd.read_pickle(f)
                    df_list.append(df)
                df_all = pd.concat(df_list, axis = 0, ignore_index=True, sort=False)
                df_all.to_pickle(f"{output_folder}/{chunk_name}{pool_name}_{filter_tag}.pkl")

            size_chunk=1000
            files_dir = f"{data_dir}/{pool_name}"
            files_list = os.listdir(files_dir)

            combine_df([f"{files_dir}/{f}" for f in files_list], "", output_dir)
            # combine_chunks(combine_df, files_dir, files_list, f"{data_dir}_{pool_name}_tmp", output_dir, size_chunk)

            with open(progress_file, "w") as f:
                f.write(str(pi+1))
            with open(f"cleanup_files.sh", "a") as f:
                f.write(f"rm -rf {files_dir}\n")

    print("run $bash cleanup_files.sh")

def calculate_demand_ratio(std_samples, hist_name="0-2+",filter_nonatomic=True,CONTRACT_DATA_FOLDER=CONTRACT_DATA_FOLDER,DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER):
    hist_filter = HISTOGRAM_BINS[hist_name]
    bin_names = list(hist_filter.keys())
    dtypes = {
        "address": 'str',
        "blockHash": 'str',
        "blockNumber":'int',
        "data":  'str',
        "logIndex": 'int',
        "removed": 'bool',
        "topics": 'object',
        "transactionHash": 'str',
        "transactionIndex": 'int',
        "poolName": 'str',
        "amount0": 'str',
        "amount1": 'object',
        "price": 'str',
        "liq": 'str',
        "tick":'str'
    }
    filter_tag = FILTERED_TAG[filter_nonatomic]
    if not os.path.isdir(DEMAND_RATIO_DATA_FOLDER):
        os.makedirs(DEMAND_RATIO_DATA_FOLDER)
    assert os.path.isdir(f"{CONTRACT_DATA_FOLDER}_{filter_tag}")
    pool_filenames = os.listdir(f"{CONTRACT_DATA_FOLDER}_{filter_tag}")
    num_pools = len(pool_filenames)

    num_swaps=[]
    ratio_data={}
    for num_samples in std_samples:
        ratio_data[num_samples] = {}
        
    for pi in range(num_pools):
        filename = pool_filenames[pi]
        pool_name = filename[:-4]
        print(f"calculate_demand_ratio {pi+1}/{num_pools} calculate pool_name {pool_name} {filter_tag}")
        df = pd.read_pickle(f"{CONTRACT_DATA_FOLDER}_{filter_tag}/{filename}")
        df = df.astype(dtypes)
        df['topics'] = df['topics'].apply(sorted).astype(str)
        dup1=df.drop_duplicates(['blockNumber','transactionIndex', 'logIndex'])
        dup2=df.drop_duplicates()
        assert dup1.equals(dup2)
        df = dup2
        df['amount1'] = df["amount1"].apply(int)
        df['txVolume'] = df["amount1"].apply(abs)
        df = df.sort_values(by=['blockNumber','transactionIndex', 'logIndex'])
        block_group = df.groupby("blockNumber")
        dfBlock = pd.DataFrame()
        dfBlock['blockDemand'] = block_group["amount1"].sum().abs()
        num_swaps.append([int(df['txVolume'].gt(0).count()),pool_name])

        def getRatio(volumeStd, blockDemand):
            if volumeStd == 0:
                return -2
            elif volumeStd > 0:
                return blockDemand / volumeStd
            elif pd.isnull(volumeStd):
                return -1
            else:
                raise ValueError

        def getBin(blockRatio):
            name = None
            for bin_name in hist_filter:
                filter_foo = hist_filter[bin_name]
                if filter_foo(blockRatio):
                    assert name is None
                    name = bin_name
            return name


        for num_samples in std_samples:
            df[f'volumdSTD{num_samples}'] = df['txVolume'].rolling(window=num_samples, center=True, min_periods=num_samples).std()
            block_group = df.groupby("blockNumber")
            dfBlock[f'volumdSTD{num_samples}'] = block_group[f"volumdSTD{num_samples}"].apply(lambda x: x.to_list()[int(len(x)/2)] if len(x) > 0 else np.nan)
            dfBlock[f'blockRatio{num_samples}'] = dfBlock.apply(lambda x: getRatio(x[f'volumdSTD{num_samples}'], x['blockDemand']), axis=1)
            dfBlock[f'poolBin{num_samples}'] = dfBlock[f'blockRatio{num_samples}'].apply(getBin)
            ratio_data[num_samples][pool_name] =  dfBlock[f'poolBin{num_samples}'].value_counts().to_dict()
        
    for num_samples in std_samples:
        json.dump(ratio_data[num_samples], open(f"{DEMAND_RATIO_DATA_FOLDER}/{num_samples}stdsamples_{filter_tag}_{hist_name}.json", "w"), indent=2)
    json.dump(num_swaps, open(f"{DEMAND_RATIO_DATA_FOLDER}/num_swaps_{filter_tag}_{hist_name}.json", "w"), indent=2)
        
def plot_demand_ratio_all_pools(hist_name="0-2+", std_samples=7000, filter_nonatomic=True, DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    if not os.path.isdir(FIGURE_FOLDER):
        os.makedirs(FIGURE_FOLDER)

    hist_data = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
    num_swaps = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/num_swaps_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
    num_swaps.sort(key=lambda x: x[0], reverse=True)
    hist_filter = HISTOGRAM_BINS[hist_name]
    non_valid_bins = ['not enough samples\nin block', '0', 'std=0']
    bin_names = list(hist_filter.keys())
    valid_bins = list(filter(lambda x: x not in non_valid_bins, bin_names))

    num_pool_groups = 5
    width = 1/(num_pool_groups+2)

    pools_bins_blocks = {}
    pools_bins_percent_all= {}
    pools_bins_percent_valid = {}
    for pool_name, pool_bins in hist_data.items():
        total_blocks = sum(pool_bins.values())
        total_valid_blocks = sum([block_count  if bin_name in valid_bins else 0 for bin_name,block_count in pool_bins.items()])

        pools_bins_blocks[pool_name] = []
        pools_bins_percent_all[pool_name] = []
        pools_bins_percent_valid[pool_name] = []

        for bin_name in bin_names:
            if bin_name in pool_bins:
                block_count = pool_bins[bin_name]
            else:
                block_count = 0     
            pools_bins_blocks[pool_name].append(block_count)       
            if total_blocks > 0:
                pools_bins_percent_all[pool_name].append(100*block_count/total_blocks)              
            if bin_name in valid_bins and total_valid_blocks > 0:
                pools_bins_percent_valid[pool_name].append(100*block_count/total_valid_blocks)
    

    fig_block = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 6))
    FIGURE_NUM+=1
    multiplier = 1
    ax_block = fig_block.subplots()
    xs = np.arange(len(bin_names))        
    num_swaps_valid = list(filter(lambda x: pools_bins_blocks[x[1]] != [], num_swaps))
    pool_group_size_valid = math.floor(len(num_swaps_valid)/num_pool_groups)
    for i in range(num_pool_groups):
        best_pools = num_swaps_valid[:pool_group_size_valid*(i+1)]
        combined_bins = [pools_bins_blocks[num_swap[1]] for num_swap in best_pools]
        combined_bins = np.average(combined_bins, axis=0)
        offset = width*multiplier
        ax_block.bar(xs + offset, combined_bins, width, label=f"top {int(100*(i+1)/num_pool_groups)}%")
        multiplier+=1
    ax_block.set_xticks(xs + .4, bin_names)
    ax_block.set_xlabel("Demand Ratio")
    ax_block.set_ylabel("Number of Blocks")
    ax_block.legend([f"top {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    ax_block.set_title(f"Number of Blocks\nDemand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")

    fig_percent = plt.figure(FIGURE_NUM, layout='constrained', figsize=(12, 6))
    FIGURE_NUM+=1
    multiplier = 1
    ax_percent = fig_percent.subplots()
    num_swaps_valid = list(filter(lambda x: pools_bins_percent_all[x[1]] != [], num_swaps))
    pool_group_size_valid = math.floor(len(num_swaps_valid)/num_pool_groups)
    for i in range(num_pool_groups):
        best_pools = num_swaps_valid[:pool_group_size_valid*(i+1)]
        combined_bins = [pools_bins_percent_all[num_swap[1]] for num_swap in best_pools]
        combined_bins = np.average(combined_bins, axis=0)
        offset = width*multiplier
        ax_percent.bar(xs + offset, combined_bins, width, label=f"top {int(100*(i+1)/num_pool_groups)}%")
        multiplier+=1
    ax_percent.set_xticks(xs + .4, bin_names)
    ax_percent.set_xlabel("Demand Ratio")
    ax_percent.set_ylabel("Percentage of Blocks")
    ax_percent.legend([f"top {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    ax_percent.set_title(f"Percentage of blocks\nDemand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")

    fig_valid = plt.figure(FIGURE_NUM, layout='constrained', figsize=(10, 6))
    FIGURE_NUM+=1
    multiplier = 1
    ax_valid = fig_valid.subplots()
    xs = np.arange(len(valid_bins))        
    num_swaps_valid = list(filter(lambda x: pools_bins_percent_valid[x[1]] != [], num_swaps))
    pool_group_size_valid = math.floor(len(num_swaps_valid)/num_pool_groups)
    for i in range(num_pool_groups):
        best_pools = num_swaps_valid[:pool_group_size_valid*(i+1)]
        combined_bins = [pools_bins_percent_valid[num_swap[1]] for num_swap in best_pools]
        combined_bins = np.average(combined_bins, axis=0)
        offset = width*multiplier
        ax_valid.bar(xs + offset, combined_bins, width, label=f"top {int(100*(i+1)/num_pool_groups)}%")
        multiplier+=1
    ax_valid.set_xticks(xs + .4, valid_bins)
    ax_valid.set_xlabel("Demand Ratio")
    ax_valid.set_ylabel("Percentage of Blocks")
    ax_valid.legend([f"top {int(100*(i+1)/num_pool_groups)}%" for i in range(num_pool_groups)])
    ax_valid.set_title(f"Percentage of valid blocks\nDemand Ratio = (block demand)/({std_samples} swaps volatility) \n{FILTERED_TAG[filter_nonatomic]}")

    fig_block.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_numblocks_{hist_name}.png")
    fig_percent.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_percentall_{hist_name}.png")
    fig_valid.savefig(f"{FIGURE_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_percentvalid_{hist_name}.png")

def make_paper_plots(DEMAND_RATIO_DATA_FOLDER=DEMAND_RATIO_DATA_FOLDER, FIGURE_FOLDER=FIGURE_FOLDER):
    global FIGURE_NUM
    if not os.path.isdir(FIGURE_FOLDER):
        os.makedirs(FIGURE_FOLDER)
    std_samples = 500
    filter_nonatomic = False
    hist_name = "0-2+"

    hist_data = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/{std_samples}stdsamples_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
    num_swaps = json.load(open(f"{DEMAND_RATIO_DATA_FOLDER}/num_swaps_{FILTERED_TAG[filter_nonatomic]}_{hist_name}.json"))
    num_swaps.sort(key=lambda x: x[0], reverse=True)
    hist_filter = HISTOGRAM_BINS[hist_name]
    non_valid_bins = ['not enough samples\nin block', '0', 'std=0']
    bin_names = list(hist_filter.keys())
    valid_bins = list(filter(lambda x: x not in non_valid_bins, bin_names))

    num_pool_groups = 1
    width = 1/(num_pool_groups+1)
    pools_bins_percent_valid = {}

    for pool_name, pool_bins in hist_data.items():
        total_valid_blocks = sum([block_count  if bin_name in valid_bins else 0 for bin_name,block_count in pool_bins.items()])
        pools_bins_percent_valid[pool_name] = []
        for bin_name in bin_names:
            if bin_name in pool_bins:
                block_count = pool_bins[bin_name]
            else:
                block_count = 0     
            if bin_name in valid_bins and total_valid_blocks > 0:
                pools_bins_percent_valid[pool_name].append(100*block_count/total_valid_blocks)

    xnames = {"(0-0.25]":'<0.25',
        "(0.25-0.50]": '0.25-0.5',
        "(0.50-0.75]": '0.5-0.75',
        "(0.75-1.0]": '0.75-1',
        "(1.0-1.25]": '1-1.25',
        "(1.25-1.5]": '1.25-1.5',
        "(1.5-1.75]": '1.5-1.75',
        "(1.75-2]": '1.75-2',
        "(2.0+)": '2 <'
        }
    fig_valid, ax_valid = plt.subplots(layout='constrained')
    fig_valid.set_figheight(5.5)

    xs = np.arange(len(valid_bins))        
    num_swaps_valid = list(filter(lambda x: pools_bins_percent_valid[x[1]] != [], num_swaps))
    pool_group_size_valid = math.floor(len(num_swaps_valid)/num_pool_groups)
    multiplier = 1
    for i in range(num_pool_groups):
        best_pools = num_swaps_valid[:pool_group_size_valid*(i+1)]
        combined_bins = [pools_bins_percent_valid[num_swap[1]] for num_swap in best_pools]
        combined_bins = np.average(combined_bins, axis=0)
        offset = width*multiplier
        ax_valid.bar(xs + offset, combined_bins, width, label=FILTERED_TAG[filter_nonatomic])
        multiplier+=1
    ax_valid.set_xlabel("Demand Ratio")

    plot_vals = []
    for b in valid_bins:
        plot_vals.append(xnames[b])
    xvals = np.arange(len(plot_vals))

    ax_valid.set_xticks(xvals + .4, plot_vals)
    ax_valid.set_ylabel("Percentage of Blocks")
    ax_valid.set_title(
f'''Figure 6: Histogram of percentage of blocks falling into various demand ratio 
ranges. The percentage are calculated for each AMM pool separately, 
and then averaged across the pools. Variability calcu-
lated over a swap window of size s = {std_samples}.''')

    ax_valid.legend(loc=(.8,.8))
    fig_valid.savefig( f"{FIGURE_FOLDER}/paper_plot_data.png")

    print(f"saved to {FIGURE_FOLDER}/paper_plot_data.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("enter data directory path")
    data_dir = sys.argv[1]
    if data_dir != "demand_ratio_paper":

        get_top_tokens()
        download_create_pool_topic()
        parse_create_pool_topic()

        download_mev_dataset()

        download_swap_topic()
        parse_swap_topic()
        combine_contract_swaps()

        calculate_demand_ratio(std_samples=[100,500,1000,7000], filter_nonatomic=True, DEMAND_RATIO_DATA_FOLDER=data_dir)
        calculate_demand_ratio(std_samples=[100,500,1000,7000], filter_nonatomic=None, DEMAND_RATIO_DATA_FOLDER=data_dir)
        calculate_demand_ratio(std_samples=[100,500,1000,7000], filter_nonatomic=False, DEMAND_RATIO_DATA_FOLDER=data_dir)


    plot_demand_ratio_all_pools(std_samples=500,filter_nonatomic=True, DEMAND_RATIO_DATA_FOLDER=data_dir)
    plot_demand_ratio_all_pools(std_samples=1000,filter_nonatomic=True, DEMAND_RATIO_DATA_FOLDER=data_dir)
    plot_demand_ratio_all_pools(std_samples=7000,filter_nonatomic=True, DEMAND_RATIO_DATA_FOLDER=data_dir)

    make_paper_plots(DEMAND_RATIO_DATA_FOLDER=data_dir)