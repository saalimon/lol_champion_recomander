import itertools
import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors
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

matchup_win_df_knn = matchup_win_df.filter(regex='T1',axis=0).filter(regex='T1',axis=1)
matchup_win_df_knn_1 = matchup_win_df_knn.copy()
# store the original dataset in 'df', and create the copy of df, df1 = df.copy().
def champion_selection_recommender(champion, num_neighbors, num_recommendation, fill="ANY"):

  number_neighbors = num_neighbors

  knn = NearestNeighbors(metric='cosine', algorithm='brute')
  knn.fit(matchup_win_df_knn.values)
  distances, indices = knn.kneighbors(matchup_win_df_knn.values, n_neighbors=number_neighbors)

  champion_index = matchup_win_df_knn.columns.tolist().index(champion)

  for m,t in list(enumerate(matchup_win_df_knn.index)):
    if matchup_win_df_knn.iloc[m, champion_index] == 0:
      sim_champions = indices[m].tolist()
      champion_distances = distances[m].tolist()
    
      if m in sim_champions:
        id_champion = sim_champions.index(m)
        sim_champions.remove(m)
        champion_distances.pop(id_champion) 

      else:
        sim_champions = sim_champions[:number_neighbors-1]
        champion_distances = champion_distances[:number_neighbors-1]
           
      champion_similarity = [1-x for x in champion_distances]
      champion_similarity_copy = champion_similarity.copy()
      nominator = 0

      for s in range(0, len(champion_similarity)):
        if matchup_win_df_knn.iloc[sim_champions[s], champion_index] == 0:
          if len(champion_similarity_copy) == (number_neighbors - 1):
            champion_similarity_copy.pop(s)
          
          else:
            champion_similarity_copy.pop(s-(len(champion_similarity)-len(champion_similarity_copy)))
            
        else:
          nominator = nominator + champion_similarity[s]*matchup_win_df_knn.iloc[sim_champions[s], champion_index]
          
      if len(champion_similarity_copy) > 0:
        if sum(champion_similarity_copy) > 0:
          predicted_r = nominator/sum(champion_similarity_copy)
        
        else:
          predicted_r = 0

      else:
        predicted_r = 0
        
      matchup_win_df_knn_1.iloc[m,champion_index] = predicted_r

  recommend_champions_list = recommend_champions(champion, num_recommendation, fill)

  return recommend_champions_list
def recommend_champions(champion, num_recommended_champions, fill):

    recommended_champions = []

    for m in matchup_win_df_knn[matchup_win_df_knn[champion] == 0].index.tolist():

        index_df = matchup_win_df_knn.index.tolist().index(m)
        predicted_rating = matchup_win_df_knn_1.iloc[index_df, matchup_win_df_knn_1.columns.tolist().index(champion)]
        recommended_champions.append((m, predicted_rating))

    sorted_rm = sorted(recommended_champions, key=lambda x:x[1], reverse=True)
    
    print('Recommended Champion(s): \n')
    rank = 1

    recommend_champions_list = []
    
    if fill == 'ANY':
        for recommended_champion in sorted_rm[:num_recommended_champions]:
            print('{}: {} - predicted winrate:{}'.format(rank, recommended_champion[0], recommended_champion[1]))
            recommend_champions_list.append(recommended_champion[0])
            rank = rank + 1
    else:
        for recommended_champion in sorted_rm[:]:
            if fill in recommended_champion[0] and rank!= num_recommended_champions:
                print('{}: {} - predicted winrate:{}'.format(rank, recommended_champion[0], recommended_champion[1]))
                rank = rank + 1
                recommend_champions_list.append(recommended_champion[0])
            elif rank > num_recommended_champions:
                break

    return recommend_champions_list
def get_best_counter_synergy_champ(current_champ_list,potential_champ_list,opponent_champ_list, top_best):

    counter_score_list = []
    synergy_score_list = []
    champ_recommended_score = {}
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


    for champ in potential_champ_list:
        temp = current_champ_list.copy()
        temp.append(champ)
        champ_counter_comp_t1 = comb_counter(temp, opponent_champ_list)
        champ_comb_t1 = combination(temp)
        counter_score_temp = get_sum_counterrate(champ_counter_comp_t1,'T1')
        counter_score_list.append(counter_score_temp)

        synergy_score_temp = get_sum_winrate(champ_comb_t1,'T1')
        synergy_score_list.append(synergy_score_temp)

    for i in range(len(synergy_score_list)):
        champ_recommended_score[potential_champ_list[i]] = (0.7*counter_score_list[i])+(0.3*synergy_score_list[i])

    sorted_score = {k: v for k, v in sorted(champ_recommended_score.items(), key=lambda item: item[1], reverse=True)}

    rank = 1
    print("\nBest champion to pick: \n")
    recommend_champions_dict = []
    for k, v in sorted_score.items():
        print("{}: {} - {}".format(rank,k, v))
        recommend_champions_dict.append({
            'rank':rank,
            'champion':k,
            'score':v
        })
        rank += 1
        if rank > top_best:
            break
    return recommend_champions_dict
def get_knn(current_champ_list, opponent_champ_list,role="ANY"):
    current_champ_list = list(filter(None, current_champ_list))
    opponent_champ_list = list(filter(None, opponent_champ_list))
    all_potential_champ_list = []
    for i in current_champ_list:
        potential_champ_list = champion_selection_recommender(i, 3, 10, role)
        all_potential_champ_list.extend(potential_champ_list)
    all_potential_champ_list = list(set(all_potential_champ_list))
    recommend_champions_dict = get_best_counter_synergy_champ(current_champ_list,all_potential_champ_list,opponent_champ_list,20)
    recommend_champions_df = pd.DataFrame(recommend_champions_dict)
    print(recommend_champions_df)
    return recommend_champions_df