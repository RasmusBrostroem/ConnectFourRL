from itertools import combinations
import pandas as pd
import testModel
from minimaxAgent import MinimaxAgent
from training import load_agent

def tournamentfunc(model_list, name_list, file_name, n_games=1000):
    # play against random
    results_list = []
    for model, model_name in zip(model_list, name_list):
        wins,ties,illegals = testModel.matchup(model, None, n_games)
        # save
        results_list.append([model_name, "random", wins/n_games, ties/n_games])


    # play against others
    all_matchups = combinations(zip(model_list, name_list), 2)
    for model, opponent in all_matchups:
        wins, ties, illegals = testModel.matchup(model[0], opponent[0], n_games)
        #save
        results_list.append([model[1], opponent[1], wins/n_games, ties/n_games])



    # play against minimax
    miniagent = MinimaxAgent()
    for model, model_name in zip(model_list, name_list):
        wins, ties, illegals = testModel.matchup(model, miniagent, n_games)
        #save
        results_list.append([model_name, "minimax", wins/n_games, ties/n_games])

    results_df = pd.DataFrame(results_list, columns=["Model", "Opponent", "Win rate", "Tie rate"])
    results_df.to_excel(file_name, index=False)


if __name__ == '__main__':
    model_list = []
    name_list = ["Defender", "RuleBoi", "AverageJoe"]
    for n in name_list:
        model_list.append(load_agent("AgentParameters", n, 14, "Small", "cpu"))

    file_name = 'testtourn2.xlsx'
    tournamentfunc(model_list, name_list, file_name, n_games=10)
    