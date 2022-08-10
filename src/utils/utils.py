import numpy as np
import pandas as pd
import re
import pickle 
import os
import matplotlib.pyplot as plt
import seaborn as sns

def process_json(file_name):
    """
    function used to read and reformat tournament json files
    Input: file_name str - path to filename
    Output: games pd.DataFrame - formatted dataframe
    """
    tmp_df = pd.read_json('{}'.format(file_name))
    games = tmp_df.explode('games')['games'].apply(pd.Series)
    games = pd.merge(games, tmp_df.drop('games',axis=1), how='left', left_index=True, right_index=True).reset_index()
    return games

def format_str_series(str_obj):
    """
    remove trailing/leading/multiple blanks and commas/dots etc
    """
    if type(str_obj) == pd.Series:
        return str_obj.str.replace("[,.;]", '', regex=True).str.lower().str.strip().str.replace('\s+', ' ', regex=True)
    else:
        return re.sub('\s+', ' ', str_obj.replace(',', '').lower().strip())

def process_cols(raw_df):
    """
    perform feature extraction and cast columns into proper format for future use
    Input: df - pd.DataFrame - raw data
    Output: df - pd.DataFrame - initial df with reformated columns and additional features
    """
    from config.definitions import ROOT_DIR

    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    df['tour_round'] = df['index'].str.rsplit('_', expand=True)[1].astype(np.int32)
    df['tour_length_days'] = (df['end_date'] - df['start_date']).dt.days
    df['tour_days_passed'] = (df['date'] - df['start_date']).dt.days
    df['tour_game'] = df['id'].str.split('_', expand=True)[2].astype(np.int32)
    df['time_control_rapid'] = np.where(df['time_control']=='rapid', 1, 0) # pd.get_dummies(df['time_control'],drop_first=True)
    df = df.sort_values(by=['name', 'date', 'tour_game'])
    df['white_game_n'] = df.groupby(['white', 'name'])['tour_game'].rank(method="first", ascending=True)
    df['black_game_n'] = df.groupby(['black', 'name'])['tour_game'].rank(method="first", ascending=True)
    if os.path.exists(os.path.join(ROOT_DIR,'outputs/helper.pickle')):
        with open(os.path.join(ROOT_DIR,'outputs/helper.pickle'), 'rb') as handle:
            name_mapping = pickle.load(handle)['name_mapping']
        df['white_latin'] = format_str_series(df['white']).replace(name_mapping)
        df['black_latin'] = format_str_series(df['black']).replace(name_mapping)
    else:
        df['white_latin'] = df['white']
        df['black_latin'] = df['black']
    
    return df

def match_missing_player_names(df_name, df_name_list, rating):
    """
    postprocessing function trying to find simmilar names since the data is very messy, threshold is set to 80% matching
    Input: df_name - str - name of the current players name from df
           df_name_list - list - list of all names available in df
           rating - df - dataframe with ratings to perform a search on
    """
    def similar(seq_a, seq_b):
        from difflib import SequenceMatcher
        return SequenceMatcher(None, seq_a, seq_b).ratio()

    pot_matches = []
    for i in rating['player_name_formatted']:
        pot_matches.append(similar(df_name, i))
    if pot_matches[np.argmax(pot_matches)]>=0.8:
        for i in range(len(pot_matches)):
            if rating['player_name_formatted'].iloc[np.argmax(pot_matches)] not in df_name_list:
                return rating[rating.columns[1]].iloc[np.argmax(pot_matches)]
            else:
                pot_matches.pop(np.argmax(pot_matches))
    return 0
    
def add_rating(raw_df, **kwargs):
    """
    add historical rating of players to a dataset and create additional features
    Input: df - pd.DataFrame - raw data
    Output: df - pd.DataFrame - initial df with reformated columns and additional features
    """

    df = raw_df.copy()
    rating_2014 = kwargs.get('rating_2014', pd.DataFrame(data=[], columns=['player_name', 'player_rank_2014', 'player_name_formatted']))
    rating_2020 = kwargs.get('rating_2020', pd.DataFrame(data=[], columns=['player_name', 'player_rank_2020', 'player_name_formatted']))

    df=pd.merge(df, rating_2014.rename(columns={'player_rank_2014':'white_rank_2014'}), how='left', left_on='white_latin', right_on='player_name_formatted').drop(['player_name_formatted'], axis=1)
    
    def create_rating_mapper(col_name, rating):
        player_names = df[(df['{}_rank_2014'.format(col_name)]==0) & (df['white_rank_2020']==0)]['{}_latin'.format(col_name)].unique()
        mapping = df[['{}_latin'.format(col_name)]].drop_duplicates()
        mapping['{}_latin_new_rank'.format(col_name)] = mapping['{}_latin'.format(col_name)].apply(lambda x: match_missing_player_names(x, player_names, rating))
        return dict(zip(mapping['{}_latin'.format(col_name)], mapping['{}_latin_new_rank'.format(col_name)]))
    
    df=pd.merge(df, rating_2020.rename(columns={'player_rank_2020':'white_rank_2020'}), how='left', left_on='white_latin', right_on='player_name_formatted').drop(['player_name_formatted'], axis=1)
    
    df=pd.merge(df, rating_2014.rename(columns={'player_rank_2014':'black_rank_2014'}), how='left', left_on='black_latin', right_on='player_name_formatted').drop(['player_name_formatted'], axis=1)
    df=pd.merge(df, rating_2020.rename(columns={'player_rank_2020':'black_rank_2020'}), how='left', left_on='black_latin', right_on='player_name_formatted').drop(['player_name_formatted'], axis=1)

    df['white_rank_2014'] = np.where(df['white_rank_2014'].isna(), df['white_rank_2014'].replace(create_rating_mapper('white', rating_2014)), df['white_rank_2014'])
    df['white_rank_2020'] = np.where(df['white_rank_2020'].isna(), df['white_rank_2020'].replace(create_rating_mapper('white', rating_2020)), df['white_rank_2020'])
    df['black_rank_2014'] = np.where(df['black_rank_2014'].isna(), df['black_rank_2014'].replace(create_rating_mapper('black', rating_2014)), df['white_rank_2014'])
    df['black_rank_2020'] = np.where(df['black_rank_2020'].isna(), df['black_rank_2020'].replace(create_rating_mapper('black', rating_2020)), df['black_rank_2020'])
    df.fillna(value={'white_rank_2014': 0, 'white_rank_2020': 0, 'black_rank_2014': 0, 'black_rank_2020': 0}, inplace=True)
    
    df['white_rank_est'] = df['white_rank_2014'] + (df['date'].dt.year-2014)*(df['white_rank_2020'] - df['white_rank_2014'])/6
    df['black_rank_est'] = df['black_rank_2014'] + (df['date'].dt.year-2014)*(df['black_rank_2020'] - df['black_rank_2014'])/6
    df['rank_diff'] = df['white_rank_est'] - df['black_rank_est']
    
    df['year'] = df['date'].dt.year
    df['white_rank_chg'] = (df['white_rank_2020'] - df['white_rank_2014'])/6
    df['black_rank_chg'] = (df['black_rank_2020'] - df['black_rank_2014'])/6

    return df.drop(['white_rank_2014', 'white_rank_2020', 'black_rank_2014', 'black_rank_2020', 'year'], axis=1).dropna()


def numeric_distribution_plot(df, target, ignore_cols=[]):
    """
    plot displot for all/selected numeric features in the dataset
    Input: df - pd.DataFrame - raw data
           target - str - target column
           ignore_cols - list - columns that will not be plotted
    Output: plt plot
    """
    num_features = df.drop(ignore_cols + [target], axis=1).select_dtypes(exclude='object').columns # only include numeric features
    print(f"There are {len(num_features)} numeric features in the dataset")
    for col in num_features: # restrics the plot to only 20 cols
        plt.figure(figsize=(12, 5))
        sns.displot(data=df, x=col, hue=target, kind='kde')
        plt.show()

def numeric_bar_plot(df, target, ignore_cols=[]):
    """
    plot bar plot + summary statistics for all/selected numeric features in the dataset
    Input: df - pd.DataFrame - raw data
           target - str - target column
           ignore_cols - list - columns that will not be plotted
    Output: plt plot
    """
    for col in df.drop(ignore_cols + [target], axis=1).select_dtypes(exclude='object').columns:
        print('variable: ', col)
        print(df[col].describe(percentiles=[.05, .25, .5, .75, 0.95]))
        df.boxplot(column=col, by=target, figsize=(6,6))
        plt.title(col)
        plt.show()