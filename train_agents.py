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
          save_benchmark_threshold=0,
          neptune_project_id=""):
    """_summary_

    Args:
        player1 (_type_): _description_
        player2 (_type_): _description_
        player1_optimizer (_type_, optional): _description_. Defaults to None.
        player2_optimizer (_type_, optional): _description_. Defaults to None.
        player1_filename (str, optional): _description_. Defaults to "player1".
        player2_filename (str, optional): _description_. Defaults to "player2".
        n_episodes (int, optional): _description_. Defaults to 2000.
        batch_size (int, optional): _description_. Defaults to 0.
        benchmarking_freq (int, optional): Number of games to play between
            benchmarks. Defaults to 100.
        benchmarking_opponents_list (list, optional): _description_.
            Defaults to [].
        benchmark_n_games (int, optional): _description_. Defaults to 20.
        save_benchmark_against (str, optional): Save . as soon as value exceeds... . Defaults to "".
        neptune_project_id (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    # TODO: Make it explicit that players' training flag needs to be set correctly
    assert len(player1_filename.split(".")) == 1,\
        "player1_filename includes file extension, please exclude it."
    assert len(player2_filename.split(".")) == 1,\
        "player2_filename includes file extension, please exclude it."
    if batch_size != 0:
        assert n_episodes % batch_size == 0,\
            "batch_size is not a multiple of n_episodes."
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
                       optimizer=player2_optimizer)
    if player2:
        player2.save_agent(file_name=player2_filename + "_final",
                           optimizer=player2_optimizer)

    if neptune_project_id:
        run.stop()
    print("Training completed.")
