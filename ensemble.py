import pandas as pd


def ensemble(sub_list, fname):
    sub_dfs = []
    for s in sub_list:
        sub_dfs.append(pd.read_csv(s))
    sub = sub_dfs[0].copy()
    sub['is_attributed'] = 0
    for s in sub_dfs:
        sub['is_attributed'] += s['is_attributed'].rank()/len(s)

    sub['is_attributed'] /= len(sub_list)
    sub.to_csv(f'sub/{fname}', index=False)


if __name__ == '__main__':
    # sub_list = ['sub/2018_05_05_04_49_03.csv',
    # 'sub/2018_05_02_15_25_21.csv']
    sub_list = ['sub/ensemble_temp.csv',
                'sub/2018_05_07_16_12_20_2018.csv',
                'sub/2018_05_07_18_23_53_3018.csv',
                'sub/2018_05_07_19_47_22_4018.csv']
    ensemble(sub_list, 'final_ensemble2.csv')
    
