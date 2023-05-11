# Playing Connect Four with Reinforcement Learning (WIP!)  

_NOTE: This repository was initially created as part of an exam project,
but has recently undergone a major refactoring.  
Currently, we're in the process of documenting the project and creating
some templates/examples of how you can extend it and implement
your own agents and training strategies. If you're interested, please come
back in a a month or so!_

## Training an agent
The script "training_script.py" serves as a basic template for learning an
agent to play the game.  
The script defines the function `train()`, which can train any two implemented
player classes against each other. Ie., you can also use the function both for
training one agent against an opponent using the Minimax-strategy.  
The script should give an idea of how to write training scripts using our
environment.

## Logging with Neptune
We use Neptune to track the training progress of agents. To use our logging
functions, you therefore need a Neptune user and write access to a project.
You also need to have your personal API token set as a system variable on your
PATH.    
See the [Neptune docs](https://docs.neptune.ai/setup/setting_api_token/) for
instructions.

## Requirements
This project was built using Python version 3.9.13.
The used packages are listed in `requirements.txt` and `req_no_cuda.txt`,
which can be used to create a virtual environment as described
[here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). 
Use `requirements.txt` if you have a CUDA-enabled GPU available for
computations and use `req_no_cuda.txt` if not.  
