"""Script for training agents.
"""
import game.players as pl
from game.Env import Env
import torch.optim as optim
import neptune


def train(player1,
          player2,
          optimizer_player1,
          optimizer_player2,
          batch_size = 20,
          n_updates= 50,
          neptune_project_id: str = ""):
    """Train players against each other and (optionally) log to neptune."""
    # Initialisation of neptune
    if neptune_project_id:
        run = neptune.init_run(project=neptune_project_id)
        player1.log_params(neptune_run=run)
        player2.log_params(neptune_run=run)

    # Initialising game environment and training variables
    environment = Env(player1, player2, allow_illegal_moves = False)
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
    # Creation of players and their optimizers
    player1 = pl.DirectPolicyAgent(player_piece=1)
    player2 = pl.DirectPolicyAgent(player_piece=-1)
    optimizer_player1 = optim.RMSprop(player1.parameters(),
                                    lr=0.1,
                                    weight_decay=0.95)
    optimizer_player2 = optim.RMSprop(player2.parameters(),
                                    lr=0.1,
                                    weight_decay=0.95)
    train(player1,
          player2,
          optimizer_player1,
          optimizer_player2,
          neptune_project_id="DLProject/ConnectFour")