"""Basic script for training 2 players against each other.

This script implements a basic training function and shows how to use our
implementation for training two agents against each other while logging the
process to Neptune and ultimately saving the resulting learned weights.

Defines the following function:
    - train: Trains 1 or 2 players using the specified optimizers, logs to the
        specified Neptune project and saves the resulting network(s).
        Returns None.

If run as __main__:
    Runs 1000 episodes of two game.players.DirectPolicyAgent players training
    simultaneously against each other, logs to the Neptune project
    "DLProject/ConnectFour" (change this for your own project) and saves the
    two resulting networks.
"""

import game.players as pl
from game.Env import Env
import torch.optim as optim
import neptune
import time


def train(player1,
          player2,
          optimizer_player1,
          optimizer_player2,
          batch_size=20,
          n_updates=50,
          benchmarking_opponents_list=[],
          benchmark_n_games=10,
          neptune_project_id: str = ""):
    """Train players against each other and (optionally) log to Neptune.

    batch_size*n_updates determines the total number of episodes played during
    the training, meaning that default number of episodes is 1000.

    Args:
        player1 (game.players.Player or subclass): Player 1 object. Must
            extend methods for game.players.Player in order to interact
            correctly with the environment, ie. can also be human or random.
        player2 (game.players.Player or subclass): Player 2 object. Must
            extend methods for game.players.Player in order to interact
            correctly with the environment, ie. can also be human or random.
        optimizer_player1: Optimizer object for player 1. Must be one of the
            classes defined in torch.optim.
        optimizer_player2: Optimizer object for player 2. Must be one of the
            classes defined in torch.optim.
        batch_size (int, optional): Number of games to play before updating
            the agent(s). Defaults to 20.
        n_updates (int, optional): Number of times to update the agent(s).
            Defaults to 50.
        neptune_project_id (str, optional): Project id for the Neptune project
            for logging. If "", does not log or initialise any Neptune run.
            Defaults to "".
    """

    # Initialisation of neptune
    if neptune_project_id:
        run = neptune.init_run(project=neptune_project_id)
        player1.log_params(neptune_run=run)
        player2.log_params(neptune_run=run)

    # Initialising game environment and training variables
    environment = Env(player1, player2, allow_illegal_moves=False)
    n_episodes = batch_size*n_updates
    start_time = time.time()
    for episode in range(1, n_episodes+1):
        environment.play_game()
        if episode % batch_size == 0:  # NOTE: Also benchmarks at batch_size
            player1.update_agent(optimizer=optimizer_player1)
            player2.update_agent(optimizer=optimizer_player2)
            if neptune_project_id:
                player1.log_stats(neptune_run=run)
                player2.log_stats(neptune_run=run)
            print(
                f"{episode+1} episodes completed. ({(episode+1)/n_episodes*100}%)"
                )
            spent_time = time.time() - start_time
            spent_per_episode = spent_time/episode
            remaining_episodes = n_episodes - episode
            remaining_time = remaining_episodes * spent_per_episode  # TODO: account for benchmarks
            print(f"Time spent: {round(spent_time/60, 1)} minutes.")
            print(f"Remaining: {round(remaining_time/60, 1)} minutes.")

            for opponent in benchmarking_opponents_list:
                # benchmark only prints if neptune_run is None.
                environment.benchmark(benchmark_player=player1,
                                      opponent=opponent,
                                      n_games=benchmark_n_games,
                                      neptune_run=run)
                environment.benchmark(benchmark_player=player2,
                                      opponent=opponent,
                                      n_games=benchmark_n_games,
                                      neptune_run=run)
    
    player1.save_agent(file_name="player1_tdtrain", optimizer=optimizer_player1)
    player2.save_agent(file_name="player2_tdtrain", optimizer=optimizer_player2)
    if neptune_project_id:
        run.stop()
    print("Training completed.")


if __name__ == "__main__":
    # Creation of players and their optimizers. Note, we could also load
    # the learned weights of previously trained agents using
    # load_network_weights of the appropriate class.
    # player1 = pl.DirectPolicyAgent(player_piece=1)
    # player2 = pl.DirectPolicyAgent(player_piece=-1)
    # optimizer_player1 = optim.RMSprop(player1.parameters(),
    #                                   lr=0.1,
    #                                   weight_decay=0.95)
    # optimizer_player2 = optim.RMSprop(player2.parameters(),
    #                                   lr=0.1,
    #                                   weight_decay=0.95)
    # Training with default parameters while logging with Neptune
    player1 = pl.TDAgent(player_piece=1)
    player2 = pl.TDAgent(player_piece=-1)
    Minimax_opp = pl.MinimaxAgent(player_piece=-1)
    Random_opp = pl.Player(player_piece=-1)
    train(player1,
          player2,
          optimizer_player1=None,
          optimizer_player2=None,
          batch_size=100,
          n_updates=100,
          benchmarking_opponents_list=[Minimax_opp, Random_opp],
          benchmark_n_games=20,
          neptune_project_id="DLProject/ConnectFour")
