
def load_and_join_raw_data(data_path):

    import numpy as np
    import pandas as pd
    import os
    import re
    import pickle
    from src.utils import utils
    from config.definitions import ROOT_DIR, RAW_DATA_PATH
    # load all files from specified folder
    df = pd.DataFrame()
    for i in os.listdir(os.path.join(ROOT_DIR, data_path)):
        df = pd.concat([df, utils.process_json(os.path.join(ROOT_DIR, data_path, i))])

    # load rating data
    rating_2014 = pd.read_csv(os.path.join(RAW_DATA_PATH,'rating_2014.txt'), sep='\t', names=['player_name','player_rank_2014'], header=None)
    rating_2014['player_name_formatted'] = utils.format_str_series(rating_2014['player_name'])
    rating_2014 = rating_2014.groupby('player_name_formatted')['player_rank_2014'].max().reset_index()

    rating_2020 = pd.read_csv(os.path.join(RAW_DATA_PATH,'rating_2020.txt'), sep='\t', names=['player_name','player_rank_2020'], header=None)
    rating_2020['player_name_formatted'] = utils.format_str_series(rating_2020['player_name'])
    rating_2020 = rating_2020.groupby('player_name_formatted')['player_rank_2020'].max().reset_index()
    # create mapping for non-latin names
    if ~os.path.exists(os.path.join(ROOT_DIR, 'outputs/helper.pickle')):
        import translators as ts
        name_mapping = {}
        for i in set(utils.format_str_series(df['white'].drop_duplicates()).to_list() + \
                    utils.format_str_series(df['black'].drop_duplicates()).to_list()):
            ch = re.findall(r'[\u4e00-\u9fff]+', i) # check if it contains non-latin characters  -  https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
            if ch:
                name_mapping[i] = utils.format_str_series(ts.google(i, from_language='zh', to_language='en')) # replace them/translate to english

        # save them for latter experiments
        with open(os.path.join(ROOT_DIR, 'outputs/helper.pickle'), 'wb') as f:
            pickle.dump({'name_mapping':name_mapping, 'rating_2014':rating_2014, 'rating_2020':rating_2020}, f)

    return df, rating_2014, rating_2020
