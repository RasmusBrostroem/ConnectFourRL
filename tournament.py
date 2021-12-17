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
    print("Start")
    model_list = []
    name_list = ["AverageJoe", "Defender", "BasicBitch", "BasicBitchV2", "TequilaBoiV2", "AverageJoeV3", "AverageJoeV4", "LastHopeBoi", "HailMaryBoi"]
    size_list = ["Small", "Small", "Mini", "Small", "Small", "Small", "Small", "Small", "Small"]
    generation_list = [44, 49, 3, 3, 49, 11, 4, 4, 1902]
    for name, size, gen in zip(name_list, size_list, generation_list):
        model_list.append(load_agent("AgentParameters", name, gen, size, "cpu"))

    file_name = 'testtourn4.xlsx'
    tournamentfunc(model_list, name_list, file_name, n_games=100)
    