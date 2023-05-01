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


def train(player1,
          player2,
          optimizer_player1,
          optimizer_player2,
          batch_size=20,
          n_updates=50,
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
    batch_size = 20
    n_updates = 50
    episodes = batch_size*n_updates
    for episode in range(episodes):
        environment.play_game()
        if episode % batch_size == 0:
            player1.update_agent(optimizer=optimizer_player1)
            player2.update_agent(optimizer=optimizer_player2)
            if neptune_project_id:
                player1.log_stats(neptune_run=run)
                player2.log_stats(neptune_run=run)
    player1.save_agent(file_name="player1", optimizer=optimizer_player1)
    player2.save_agent(file_name="player2", optimizer=optimizer_player2)
    run.stop()
    print("Training completed.")


if __name__ == "__main__":
    # Creation of players and their optimizers. Note, we could also load
    # the learned weights of previously trained agents using
    # load_network_weights of the appropriate class.
    player1 = pl.DirectPolicyAgent(player_piece=1)
    player2 = pl.DirectPolicyAgent(player_piece=-1)
    optimizer_player1 = optim.RMSprop(player1.parameters(),
                                      lr=0.1,
                                      weight_decay=0.95)
    optimizer_player2 = optim.RMSprop(player2.parameters(),
                                      lr=0.1,
                                      weight_decay=0.95)
    # Training with default parameters while logging with Neptune
    train(player1,
          player2,
          optimizer_player1,
          optimizer_player2,
          neptune_project_id="DLProject/ConnectFour")
