import game.players as pl
from game.Env import Env
import neptune
import time


def self_train(player1,
               n_episodes=1000,
               benchmarking_freq=200,
               neptune_project_id: str = "",
               file_name="player1_selftrain"):

    # Initialisation of neptune
    if neptune_project_id:
        run = neptune.init_run(project=neptune_project_id)
        player1.log_params(neptune_run=run)

    environment = Env(player1, None, allow_illegal_moves=False)
    start_time = time.time()
    for episode in range(n_episodes):
        environment.self_play()
        if episode % benchmarking_freq == 0:
            # benchmark and figure out how to log it
            if neptune_project_id:
                player1.log_stats(neptune_run=run)
            print(
                f"{episode+1} episodes completed. ({(episode+1)/n_episodes*100}%)"
                )
            spent_time = time.time() - start_time
            spent_per_episode = spent_time/(episode+1)
            remaining_episodes = n_episodes - episode
            remaining_time = remaining_episodes * spent_per_episode
            print(f"Time spent: {round(spent_time/60, 1)} minutes.")
            print(f"Remaining: {round(remaining_time/60, 1)} minutes.")
    player1.save_agent(file_name=file_name, optimizer=None)
    if neptune_project_id:
        run.stop()
    print("Training completed.")


if __name__ == "__main__":
    player1 = pl.TDAgent(player_piece=1)
    player1.load_network_weights("learned_weights/player1_selftrain.pt")
    self_train(player1,
               n_episodes=10000,
               file_name="player1_selftrain_11k")
