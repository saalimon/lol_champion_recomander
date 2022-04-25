import pickle
import itertools
import pandas as pd
import numpy as np
def pred(top_t1,top_t2,mid_t1,mid_t2,adc_t1,adc_t2,jg_t1,jg_t2,sup_t1,sup_t2):
    matchup_df = pd.read_csv("./model/matchup_df.csv", index_col=0)
    matchup_df_all = pd.read_csv("./model/matchup_df_all.csv", index_col=0)
    matchup_lose = pd.read_csv("./model/matchup_df_lose.csv", index_col=0)
    for i in range(1,11):
        matchup_df_all.replace(i, np.nan, inplace=True)
    matchup_win_df = matchup_df/matchup_df_all
    matchup_win_df.fillna(0, inplace=True)
    matchup_lose_df = matchup_lose/matchup_df_all
    matchup_lose_df.fillna(0, inplace=True)
    matchup_win_all_df = matchup_win_df.copy()
    matchup_lose_all_df = matchup_lose_df.copy()
    matchup_win_T1_df = matchup_win_all_df.filter(regex='T1',axis=0).filter(regex='T1',axis=1)
    matchup_win_T2_df = matchup_lose_all_df.filter(regex='T2',axis=0).filter(regex='T2',axis=1)
    matchup_lose_T1_df = matchup_win_all_df.filter(regex='T1',axis=0).filter(regex='T2',axis=1)
    matchup_lose_T2_df = matchup_lose_all_df.filter(regex='T2',axis=0).filter(regex='T1',axis=1)

    def get_score_for_pred(prediction_champ, pred_list):
        def combination(champs):
            return_list = []
            for L in range(0, len(champs)+1):
                for subset in itertools.combinations(champs, L):
                    if len(subset) == 2:
                        return_list.append(subset)
            return return_list
        def comb_counter(champs1, champs2):
            return_list = list(itertools.product(champs1,champs2))
            return return_list
        def get_sum_winrate(champ_comb, team):
            result_wr_score = 0
            for i in champ_comb:
                win_rate = find_winrate(i[0],i[1],team)
                result_wr_score += win_rate
            return result_wr_score
        def get_sum_counterrate(champ_comb, team):
            result_cr_score = 0
            for i in champ_comb:
                counter_rate = find_counterrate(i[0],i[1],team)
                result_cr_score += counter_rate
            return result_cr_score
        def find_winrate(champ1,champ2,team):
            if team == 'T1':
                return matchup_win_T1_df.filter(regex=champ1,axis=0)[champ2].values[0]
            else:
                return matchup_win_T2_df.filter(regex=champ1,axis=0)[champ2].values[0]
        def find_counterrate(champ1,champ2,team):
            if team == 'T1':
                return matchup_lose_T1_df.filter(regex=champ1,axis=0)[champ2].values[0]
            else:
                return matchup_lose_T2_df.filter(regex=champ1,axis=0)[champ2].values[0]


        prediction_champ['blue_team_synergy'] = 0
        prediction_champ['red_team_synergy'] = 0
        prediction_champ['blue_to_red_team_synergy_diff'] = 0

        prediction_champ['blue_team_counter'] = 0
        prediction_champ['red_team_counter'] = 0

        team1 = [champ for champ in pred_list if 'T1' in champ]
        team2 = [champ for champ in pred_list if 'T2' in champ]
    
        champ_comb_t1 = combination(team1)
        champ_comb_t2 = combination(team2)
        champ_counter_comp_t1 = comb_counter(team1, team2)
        champ_counter_comp_t2 = comb_counter(team2, team1)
        synergy_score_t1 = get_sum_winrate(champ_comb_t1,'T1')
        synergy_score_t2 = get_sum_winrate(champ_comb_t2,'T2')
        counter_score_t1 = get_sum_counterrate(champ_counter_comp_t1,'T1')
        counter_score_t2 = get_sum_counterrate(champ_counter_comp_t2,'T2')

        prediction_champ['blue_team_synergy'] = synergy_score_t1
        prediction_champ['red_team_synergy'] = synergy_score_t2
        prediction_champ['blue_to_red_team_synergy_diff'] = prediction_champ['blue_team_synergy'] - prediction_champ['red_team_synergy']
        prediction_champ['blue_team_counter'] = counter_score_t1
        prediction_champ['red_team_counter'] = counter_score_t2

        return prediction_champ

    current_champ_list = [top_t1, mid_t1, adc_t1, jg_t1, sup_t1]
    opponent_champ_list = [top_t2, mid_t2, adc_t2, jg_t2, sup_t2]
    pred_list = current_champ_list+opponent_champ_list
    with open('./model/columns_all.pkl', 'rb') as f:
        col = pickle.load(f)
    print(len(col))
    pred_dict = {}
    for i in col:
        if i in pred_list:
            pred_dict[i] = 1
        else:
            pred_dict[i] = 0
    prediction = pd.Series(pred_dict)
    prediction = get_score_for_pred(prediction, pred_list)
    prediction = prediction.values.reshape(1,1545)
    with open('./model/clf_gb_blue.sav', 'rb') as f:
        clf_gb_blue = pickle.load(f)
    pred_result = clf_gb_blue.predict_proba(prediction)
    return pred_result
