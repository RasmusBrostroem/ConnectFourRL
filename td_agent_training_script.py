import game.players as pl
from game.Env import Env
import neptune
import time


def self_train(player1,
               n_episodes=1000,
               benchmarking_freq=200,
               benchmarking_opponents_list=[],
               benchmark_n_games=10,
               neptune_project_id: str = "",
               file_name="player1_selftrain"):

    # Initialisation of neptune
    if neptune_project_id:
        run = neptune.init_run(project=neptune_project_id)
        player1.log_params(neptune_run=run)
    else:
        run = None
    environment = Env(player1, None, allow_illegal_moves=False)
    start_time = time.time()
    for episode in range(1, n_episodes+1):
        environment.self_play()
        if episode % benchmarking_freq == 0:
            if neptune_project_id:
                player1.log_stats(neptune_run=run)
            print(
                f"{episode+1} episodes completed. ({(episode+1)/n_episodes*100}%)"
                )
            spent_time = time.time() - start_time
            spent_per_episode = spent_time/episode
            remaining_episodes = n_episodes - episode
            remaining_time = remaining_episodes * spent_per_episode
            print(f"Time spent: {round(spent_time/60, 1)} minutes.")
            print(f"Remaining: {round(remaining_time/60, 1)} minutes.")

            for opponent in benchmarking_opponents_list:
                # benchmark only prints if neptune_run is None.
                environment.benchmark(benchmark_player=player1,
                                      opponent=opponent,
                                      n_games=benchmark_n_games,
                                      neptune_run=run)
    player1.save_agent(file_name=file_name, optimizer=None)
    if neptune_project_id:
        run.stop()
    print("Training completed.")


if __name__ == "__main__":
    player1 = pl.TDAgent(player_piece=1)
    Minimax_opp = pl.MinimaxAgent(player_piece=-1)
    Random_opp = pl.Player(player_piece=-1)
    self_train(player1=player1,
               n_episodes=100,
               benchmarking_freq=30,
               benchmark_n_games=10,
               benchmarking_opponents_list=[Minimax_opp, Random_opp],
               neptune_project_id="DLProject/ConnectFour",
               file_name="player1_selftrain")
