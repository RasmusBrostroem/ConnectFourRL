import os
import pandas as pd
import gym
import gym_game
import neptune.new as neptune
import pygame as pg
import torch
import torch.optim as optim
import numpy as np
from agent import DirectPolicyAgent, DirectPolicyAgent_large, DirectPolicyAgent_mini
from training import train_agent

Excel_file_name = "Loss_mean_agents.csv"
models = pd.read_csv(Excel_file_name, sep=",")
#models["IllegalMove"] = models["IllegalMove"].map({"False": False, "True": True})
#models["MinMax"] = models["MinMax"].map({"False": False, "True": True})

pg.init()

env = gym.make('ConnectFour-v0')

for i, model in models.iterrows():
    if not pd.isnull(model["Neptune"]):
        continue

    run = neptune.init(project="DLProject/ConnectFour")

    # Agent
    if model["AgentSize"] == "Small":
        device = "cpu"
        agent = DirectPolicyAgent(device, model["RewardDecay"])
    elif model["AgentSize"] == "Mini":
        device = "cpu"
        agent = DirectPolicyAgent_mini(device, model["RewardDecay"])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = DirectPolicyAgent_large(device, model["RewardDecay"])
    
    if not pd.isna(model["PreTrain"]):
        pre_train_path = os.path.join("AgentParameters", model["PreTrain"])
        agent = torch.load(pre_train_path)
        print(f"Loading pretrained network for: {pre_train_path}")

    agent.to(device)
    
    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=model["LearningRate"], weight_decay=model["WeightDecay"])

    # Environment
    env.configureRewards(win = model["WinReward"], loss = model["LossReward"], tie = model["TieReward"], illegal = model["IllegalReward"])
    env.reset()

    # Neptune Params
    params = {"generations": model["Generations"],
          "episodes_per_gen": model["Episodes"],
          "batch_size": model["BatchSize"],
          "optimizer": "Adam",
          "learning_rate": model["LearningRate"],
          "weight_decay": model["WeightDecay"],
          "reward_decay": agent.gamma,
          "win_reward": env.game.win,
          "loss_reward": env.game.loss,
          "tie_reward": env.game.tie,
          "illegal_reward": env.game.illegal,
          "illegal_move_possible": model["IllegalMove"],
          "MiniMax": model["MinMax"]}

    run["parameters"] = params

    for name, param in agent.named_parameters():
        run["model/summary"].log(f"name: {name} with size: {param.size()}")

    train_agent(env, agent, optimizer, run, model["Generations"], model["Episodes"], model["BatchSize"], model["MinMax"], model["AgentSize"], model["IllegalMove"], device, ["AgentParameters", model["ModelName"]], show_every=1000000)

    models.loc[i, "Neptune"] = run._short_id

    run.stop()
    models.to_csv(Excel_file_name, index=False)