"""Define the logic for training against other player or through self-play.

This script implements a training function which can be used for the following
training strategies:
  - let 2 players play against each other, both of them learning
  - let 2 players play against each other, with only player1 learning. The
    opponent can be any Player object.
  - let 1 player train by playing against itself.
The training function optionally logs to neptune and saves the resulting
learned weights.

Defines the following function:
    - train: Trains 1 or 2 players using the specified optimizers, logs to the
        specified Neptune project and saves the resulting network(s).
        Returns None.

If run as __main__:
    Runs 2000 episodes of 1 players.TDAgent in self-play, benchmarking at each
    100 games by playing 20 games against both a random player and a minimax
    player. Logs to the Neptune project "DLProject/ConnectFour" (change this
    for your own project) and saves the network if the win-rate against the
    minimax opponent exceeds 0.4, as well as at the end of the training.
"""
import game.players as pl
from game.Env import Env
import neptune
import time


def train(player1,
          player2,
          player1_optimizer=None,
          player2_optimizer=None,
          player1_filename="player1",
          player2_filename="player2",
          n_episodes=2000,
          batch_size=0,
          benchmarking_freq=100,
          benchmarking_opponents_list=[],
          benchmark_n_games=20,
          save_benchmark_against="",
          save_benchmark_threshold=0.0,
          neptune_project_id=""):
    """Train player1 against player2 (or self-play), benchmark, save results.

    Can benchmark against several player classes, but saving based on
    performance is only implemented towards one of the opponents. This
    opponent should be the only one of its class, as it is the class name
    (provided with save_benchmark_against) which determines the saving based
    on the win-rate.
    The .training attribute of the player objects must be set correctly prior
    to training. Player2 will not get updated or benchmarked if
    player2.training=False.


    Args:
        player1 (game.players.Player or subclass): Player 1 object. Must
         extend methods for game.players.Player in order to interact
         correctly with the environment, ie. can also be human or random.
        player2 (game.players.Player or subclass or None): Player 2 object.
         Must extend methods for game.players.Player in order to interact
         correctly with the environment, ie. can also be human or random.
         If None, player1 will be trained in self-play using Env.self_play().
       optimizer_player1: Optimizer object for player 1. Must be one of the
        classes defined in torch.optim or None.
        optimizer_player2: Optimizer object for player 2. Must be one of the
         classes defined in torch.optim or None. Unused if player2=None.
        player1_filename (str, optional): Filename to use for saving network
         weights of player1. Do not include file extension. If saved after a
         benchmark during training, it will be changed following the logic in
         Env.benchmark(). When saving at the end of training, "_final" will be
         added to the file name. Defaults to "player1".
        player2_filename (str, optional): Filename to use for saving network
         weights of player2. Unused if player2=None, but apart from that its
         usage is the same as player1_filename. Defaults to "player2".
        n_episodes (int, optional): Total number of games to play.
         Defaults to 2000.
        batch_size (int, optional): Number of games to play in between
         updates. If the agent(s) to train are implemented with incremental
         updates (e.g. as TDAgent), then set this to 0. Defaults to 0.
        benchmarking_freq (int, optional): Number of games to play between
         benchmarks. Benchmarking should occur immediately after network
         updates, and hence should be a multiple of batch_size unless this is
         0. Defaults to 100.
        benchmarking_opponents_list (list, optional): List of player objects
         to benchmark against. Defaults to [].
        benchmark_n_games (int, optional): Number of games to play against
         each of the benchmarking opponents. Defaults to 20.
        save_benchmark_against (str, optional): Class name of one of the
         benchmarking opponnents. The win-rate against this opponent will then
         be tracked and network weights will be saved when performance
         improves. If "", does not save. Defaults to "".
        save_benchmark_threshold (float, optional): Minimal win-rate required
         to save network weights based on benchmarking performance. Once the
         minimal win-rate has been exceeded, the network will only be saved
         when it exceeds the previously best performance. Defaults to 0.0.
        neptune_project_id (str, optional): Project id for the Neptune project
         for logging. If "", does not log or initialise any Neptune run.
         Defaults to "".

    Returns:
        None
    """
    assert len(player1_filename.split(".")) == 1,\
        "player1_filename includes file extension, please exclude it."
    assert len(player2_filename.split(".")) == 1,\
        "player2_filename includes file extension, please exclude it."
    if batch_size != 0:
        assert n_episodes % batch_size == 0,\
            "n_episodes is not a multiple of batch_size."
        assert benchmarking_freq % batch_size == 0,\
            "benchmarking_freq needs to be a multiple of batch_size."

    def print_status(episode_i,
                     n_episodes,
                     start_time) -> None:
        print(f"{episode_i} episodes completed of {n_episodes}.")
        spent_time = time.time() - start_time
        spent_per_episode = spent_time/episode_i
        remaining_episodes = n_episodes - episode_i
        remaining_time = remaining_episodes * spent_per_episode
        print(f"Time spent: {round(spent_time/60, 1)} minutes.")
        print(f"Remaining: {round(remaining_time/60, 1)} minutes.")
        return None

    if neptune_project_id:
        run = neptune.init_run(project=neptune_project_id)
        player1.log_params(neptune_run=run)
        if player2:
            player2.log_params(neptune_run=run)
    else:
        run = None

    environment = Env(player1, player2, allow_illegal_moves=False)

    if batch_size == 0:
        checkpoint_time = benchmarking_freq
    else:
        checkpoint_time = batch_size
    start_time = time.time()
    for episode in range(1, n_episodes+1):
        if player2:
            environment.play_game()
        else:
            environment.self_play()
        if episode % checkpoint_time == 0:
            if neptune_project_id:
                player1.log_stats(neptune_run=run)
                if player2:
                    player2.log_stats(neptune_run=run)

            player1.update_agent(optimizer=player1_optimizer)
            if player2 and player2.training:
                player2.update_agent(optimizer=player2_optimizer)
            print_status(episode_i=episode,
                         n_episodes=n_episodes,
                         start_time=start_time)
            if episode % benchmarking_freq == 0:
                print("Benchmarking...")
                for opponent in benchmarking_opponents_list:
                    environment.benchmark(
                        benchmark_player=player1,
                        opponent=opponent,
                        n_games=benchmark_n_games,
                        benchmark_player_name=player1_filename,
                        benchmark_player_optim=player1_optimizer,
                        save_threshold=save_benchmark_threshold,
                        save_against=save_benchmark_against,
                        neptune_run=run)
                    if player2 and player2.training:
                        environment.benchmark(
                            benchmark_player=player2,
                            opponent=opponent,
                            n_games=benchmark_n_games,
                            benchmark_player_name=player2_filename,
                            benchmark_player_optim=player2_optimizer,
                            save_threshold=save_benchmark_threshold,
                            save_against=save_benchmark_against,
                            neptune_run=run)
                print("Benchmarking done, resuming training.")

    player1.save_agent(file_name=player1_filename + "_final",
                       optimizer=player1_optimizer)
    if player2:
        player2.save_agent(file_name=player2_filename + "_final",
                           optimizer=player2_optimizer)

    if neptune_project_id:
        run.stop()
    print("Training completed.")


if __name__ == "__main__":
    player1 = pl.TDAgent(player_piece=1,
                         epsilon=0.1,
                         alpha=0.01,
                         loss_reward=-0.1)
    player2 = pl.MinimaxAgent(player_piece=-1, training=False)
    Minimax_opp = pl.MinimaxAgent(player_piece=-1)
    Random_opp = pl.Player(player_piece=-1)
    train(player1=player1,
          player2=player2,
          player1_filename="TD_v_minimax_2k",
          benchmarking_opponents_list=[Minimax_opp, Random_opp],
          save_benchmark_against="MinimaxAgent",
          save_benchmark_threshold=0.4,
          neptune_project_id="")
