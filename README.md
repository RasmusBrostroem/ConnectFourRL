# Playing Connect Four with Reinforcement Learning (WIP!)  

_NOTE: This repository was initially created as part of an exam project,
but has recently undergone a major refactoring.  
Currently, we're in the process of documenting the project and creating
some templates/examples of how you can extend it and implement
your own agents and training strategies. If you're interested, please come
back in a a month or so!_

# Proposed sections for this readme

## Training an agent
The script "training_script.py" serves as a basic template for learning an
agent to play the game.  
The script defines the function `train()`, which can train any two implemented
player classes against each other. Ie., you can also use the function both for
training one agent against an opponent using the Minimax-strategy.  
The script should give an idea of how to write training scripts using our
environment.

## Requirements
This project was built using Python version 3.9.13.
The used packages are listed in `requirements.txt`, which can be used to
create a virtual environment as described
[here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).  
