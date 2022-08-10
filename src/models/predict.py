from config.definitions import ROOT_DIR, RAW_DATA_PATH
def predict_new_cases(model_name='minmax_te', data_path='data'):
    from src.utils import utils
    from src.data.load_raw_data import load_and_join_raw_data
    import pickle
    import os
    from config.definitions import ROOT_DIR

    df, rating_2014, rating_2020 = load_and_join_raw_data(os.path.join(ROOT_DIR, data_path))
    df_processed = df.pipe(utils.process_cols).pipe(utils.add_rating, **{'rating_2014':rating_2014, 'rating_2020':rating_2020})

    # load transformer
    with open(os.path.join(ROOT_DIR, 'models/{}_model.pickle'.format(model_name)), 'rb') as handle:
        model_obj = pickle.load(handle)
    df_processed['prediction'] = model_obj['model'].predict(df_processed[model_obj['features']]) / 2
    
    return df_processed

def update_json_w_pred(df, rel_path):
    # name, index
    import os
    import json
    for nm in df['name'].unique():
        with open(os.path.join(RAW_DATA_PATH, rel_path, '{}.json'.format(nm)), 'r', encoding="utf8") as f:
            test = json.load(f)
        for idx in df[df['name']==nm]['index'].unique():
            updated_val = []
            for i in test['games'][idx]:
                i['result'] = df[df['id']==i['id']]['prediction'].values[0]
                updated_val.append(i)
            test['games'][idx] = updated_val
        with open(os.path.join(RAW_DATA_PATH, rel_path, '{}.json'.format(nm)), 'w', encoding="utf8") as f:
            test = json.dump(test, f)