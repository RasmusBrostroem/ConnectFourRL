"""This module provides agent classes for playing and learning the game.

Defines the following classes:
    - Player(): Acts as a template which the other classes extends. All
        player classes need to extend this class in order to have the methods
        required by the other scripts.
        Places pieces randomly when playing.
    - DirectPolicyAgent(): Extends Player() with a neural network deciding
        which pieces to place. Also includes functionality for loading/
        updating/saving the network.
    - DirectPolicyAgent_large(): Same as the prior, with a larger network.
    - DirectPolicyAgent_small(): Same as the prior, with a smaller network.
    - DirectPolicyAgent_mini(): Same as the prior, with a very small network.
    - HumanPlayer(): Allows the user to play the game through console input.
    - MinimaxAgent(): Implements the minimax algorithm for Connect Four.
        Uses this to decide where to place pieces.

This module has several dependencies. We recommend creating a virtual
environment from our requirements.txt file to ensure compatibility and optimal
performance. Please refer to the README.md for instructions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import neptune
from os import path, mkdir
import json
import git
from game.connectFour import connect_four


class Player():
    """Represents a player in Connect Four playing randomly. Template class.

    The class provides methods for selecting a column to place a piece in,
    updating statistics and rewards for the player, and logging performance
    metrics to Neptune.
    The class acts as a template class and contains all the methods and
    attributes a player class should provide to correctly interact with our
    implementation of Connect Four. Our implemented environment assumes that
    players provide the attributes and methods contained in this template.

    Attributes:
        Player configuration (these can be specified at initialisation):
            params: Parameters relevant for training. See __init__ docstring.
            playerPiece: Value of the player's piece in matrix representation.
            device: The device on which to perform computations (eg "cuda").

        Logging:
            neptune_id: Neptune id of the current training run.
            stats: Dict of performance metrics for the current training run.
            total_games: Total number of games played by the agent (across
                all training runs).
            probs: List of probability for each chosen move.

        Updating/learning:
            saved_log_probs: List of log(probability) for each chosen move.
            rewards: List of rewards received for each move in the episode.
            gamma: Discount factor used for calculating rewards.

    Methods:
        select_action(game, legal_moves=[]): Decide (randomly) where to place
            the next piece.
        calculate_rewards(): Calculates discounted rewards at end of episode.
        update_agent(optimizer=None): Placeholder for updating network
            weights.
        load_network_weights(*args, **kwargs): Placeholder method that can be
            overridden to load the a trained agent.
        save_agent(*args, **kwargs): Placeholder method that can be overridden
            to save a trained agent.
        update_stats(): Update performance metrics based on the outcome of an
            episode.
        log_params(neptune_run): Log player parameters to Neptune experiment.
        log_stats(neptune_run): Log performance metrics to a Neptune
            experiment.


    When extending the class, it will usually be sufficient to overwrite the
    following methods:
        - select_action,
        - update_agent,
        - load_network_weights and save_agent.
    """

    def __init__(self, player_piece: int, **kwargs) -> None:
        """Initialise a new Player object.

        Args:
            player_piece (int): The value of the player's piece (1 or -1).

        Keyword arguments:
            win_reward (float, default=1): The reward value received when
                winning a game.
            loss_reward (float, default=-1): The reward value received when
                losing a game.
            tie_reward (float, default=0.5): The reward value received when
                tying a game.
            illegal_reward (float, default=-5): The reward value received when
                attempting an illegal move (if doing so is allowed).
            not_ended_reward (float, default=0): The reward value received
                at each step when game has not ended.
            gamma (float, default=0.8): The discount factor used when
                calculating rewards.
            device (str, default="cpu"): The device on which to perform
                computations, e.g. "cpu" or "cuda".
        """
        self.params = {
            "win_reward": 1,
            "loss_reward": -1,
            "tie_reward": 0.5,
            "illegal_reward": -5,
            "not_ended_reward": 0,
            "gamma": 0.8,
            "device": "cpu"
        }
        self.params.update(kwargs)

        self.playerPiece = player_piece
        self.device = self.params["device"]

        # Parameters used for logging to neptune
        self.neptune_id = ""
        self.stats = {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "illegals": 0,
            "games": 0,
            "probs_success_sum": 0,
            "moves_success_total": 0,
            "probs_failure_sum": 0,
            "moves_failure_total": 0,
            "loss_sum": 0
        }
        self.total_games = 0
        self.probs = []

        # Parameters used for updating agent
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = self.params["gamma"]

    def select_action(self,
                      game: connect_four,
                      illegal_moves_allowed: bool = True) -> int:
        """Choose a random valid column to place the next piece in.

        Args:
            game (connect_four): Current connect_four game instance.
            illegal_moves_allowed (bool, optional): bool denoting whether
                illegal moves are allowed or not. Only provided
                for extendability, as this method does not actually need the
                argument but subclasses might. Defaults to True.

        Returns:
            int: Index of the chosen column (0-indexed).

        This method should be overridden by subclasses.
        """
        return random.choice(game.legal_cols())

    def calculate_rewards(self) -> None:
        """Calculate the discounted rewards for each move the player made.

        This method should be called at the end of an episode.

        Returns: None.
        """
        final_reward = self.rewards[-1]
        for i, val in enumerate(reversed(self.rewards)):
            if val != 0 and i != 0:
                break

            weighted_reward = self.gamma**i * final_reward
            self.rewards[len(self.rewards)-(i+1)] = weighted_reward

    def update_agent(self, optimizer=None) -> None:
        """Placeholder method for updating agent parameters and resetting.

        Args:
            optimizer (optional): An optimizer object used to update neural
                network weights. Only relevant for certain subclasses.
                Defaults to None.

        Returns: None.

        Note: When overwriting this method, you need to delete the elements of
        self.rewards and self.saved_log_probs for the purposes of reward
        calculation and logging. Failing to do so will lead to problems,
        unless you also change the other methods.
        """
        # Important: delete lists after update call
        del self.rewards[:]
        del self.saved_log_probs[:]

    def load_network_weights(self, *args, **kwargs) -> None:
        """Placeholder, ensures that the method can be called for all agents.

        This method should be overridden by subclasses, such that relevant
        trained parameters stored by self.save_agent() can be loaded.
        See the DirectPolicyAgent() subclass for an example of such extension.
        """
        pass

    def save_agent(self, *args, **kwargs) -> None:
        """Placeholder, ensures that the method can be called for all agents.

        This method should be overridden by subclasses, such that relevant
        trained parameters are stored in a loadable format.
        See the DirectPolicyAgent() subclass for an example of such extension.
        """
        pass

    def update_stats(self) -> None:
        """Update metrics with the outcome of a game and delete self.probs."""
        final_reward = self.rewards[-1]

        self.stats["games"] += 1
        self.total_games += 1
        if final_reward == self.params["loss_reward"]:
            self.stats["losses"] += 1
        elif final_reward == self.params["win_reward"]:
            self.stats["wins"] += 1
        elif final_reward == self.params["tie_reward"]:
            self.stats["ties"] += 1
        elif final_reward == self.params["illegal_reward"]:
            self.stats["illegals"] += 1

        if final_reward == self.params["loss_reward"] or \
                final_reward == self.params["illegal_reward"]:
            self.stats["probs_failure_sum"] += sum(self.probs)
            self.stats["moves_failure_total"] += len(self.probs)
        else:
            self.stats["probs_success_sum"] += sum(self.probs)
            self.stats["moves_success_total"] += len(self.probs)

        del self.probs[:]

    def log_params(self, neptune_run: neptune.Run) -> None:
        """Log player parameters in self.params to Neptune.

        Args:
            neptune_run (neptune.Run): Instance of the current neptune run.
        """
        self.neptune_id = neptune_run["sys/id"].fetch()
        neptune_run[f"player{self.playerPiece}/params"] = self.params

    def log_stats(self, neptune_run: neptune.Run) -> None:
        """Log player metrics from self.stats to Neptune and reset self.stats.

        In addition to win/loss/tie/illegal-rates, the following is stored:
            averagePropSuccess: The average probability for the chosen moves
                in games which ended in the agent winning or tying.
            averagePropFailure: The average probability for the chosen moves
                in games which ended in the agent losing or making an illegal
                move.

        Args:
            neptune_run (neptune.Run): Instance of the current neptune run.
        """
        folder_name = f"player{self.playerPiece}/metrics"
        neptune_run[folder_name + "/winrate"].log(
            self.stats["wins"]/self.stats["games"])
        neptune_run[folder_name + "/lossrate"].log(
            self.stats["losses"]/self.stats["games"])
        neptune_run[folder_name + "/tierate"].log(
            self.stats["ties"]/self.stats["games"])
        neptune_run[folder_name + "/illegalrate"].log(
            self.stats["illegals"]/self.stats["games"])

        neptune_run[folder_name + "/loss_sum"].log(self.stats["loss_sum"])
        try:
            neptune_run[folder_name + "/averagePropSuccess"].log(
                self.stats["probs_success_sum"] /
                self.stats["moves_success_total"])
            neptune_run[folder_name + "/averagePropFailure"].log(
                self.stats["probs_failure_sum"] /
                self.stats["moves_failure_total"])
        except ZeroDivisionError:
            pass

        self.stats = dict.fromkeys(self.stats, 0)  # Sets all values back to 0


class DirectPolicyAgent(nn.Module, Player):
    """Extends Player() to create an agent using a neural network for playing.

    The agent uses a neural network to output probabilities for each column
    and trains using a Direct Policy Gradient method.
    The class extends the template Player() class and overwrites the following
        - select_action,
        - update_agent,
        - save_agent,
        - load_network_weights.
    The following attributes are provided in addition to those in Player(),
        Layers in the neural network: L1, L2, L3, L4 and final.
    as well as the method forward(x), which calculates the agent's decision
    given a game state.

    This class can serve as a template for agents using neural networks and
    direct policy - see, for instance, our own extension
    DirectPolicyAgent_mini and DirectPolicyAgent_large which simply define a
    smaller and larger network structure, respectively.
    """
    def __init__(self, **kwargs):
        """Construct an agent with random weights. See Player() for kwargs."""
        Player.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.L1 = nn.Linear(42, 200)
        self.L2 = nn.Linear(200, 300)
        self.L3 = nn.Linear(300, 100)
        self.L4 = nn.Linear(100, 100)
        self.final = nn.Linear(100, 7)

    def forward(self, x):
        """Pass a game state through the network to decide which move to make.

        Args:
            x (Tensor): Flattened matrix representation of game state.

        Returns:
            Tensor: Probability for each column. The final layer is softmax,
                so the output tensor sums to 1.
        """
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.L3(x)
        x = F.relu(x)
        x = self.L4(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)

    def select_action(self,
                      game: connect_four,
                      illegal_moves_allowed: bool = True):
        """Choose placement of next piece given game and illegal move rule.

        Args:
            game (connect_four): Current connect_four game object.
            illegal_moves_allowed (bool): bool indicating whether or not
                illegal moves are allowed. Defaults to True.

        Returns:
            Tensor: Index of the chosen column.
        """
        board = game.return_board() * self.playerPiece
        board_vector = torch.from_numpy(board).float().flatten()
        board_vector = board_vector.to(self.device)
        probs = self.forward(board_vector)
        move = Categorical(probs.to("cpu"))
        action = move.sample()
        if not illegal_moves_allowed and action not in game.legal_cols():
            action = torch.tensor(random.choice(game.legal_cols()))

        self.saved_log_probs.append(move.log_prob(action))
        self.probs.append(probs[action].detach().numpy())
        return action.to("cpu")

    def update_agent(self, optimizer) -> None:
        """Update network parameters by backpropagating using the optimizer.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer object for training.
        """
        loss = [-log_p * r for log_p, r in zip(self.saved_log_probs,
                                               self.rewards)]

        loss = torch.stack(loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]
        # save loss to agent for logging
        self.stats["loss_sum"] += loss.detach().numpy()

    def save_agent(self,
                   file_name: str,
                   optimizer=None,
                   folder: str = "learned_weights",
                   store_metadata: bool = True) -> None:
        """Save learnable parameters, optimizer dictionary and metadata.

        Save data relevant for inference and resuming training with
        torch.save(). Overrides Player.save_agent().
        A dictionary is stored at "folder/file_name.pt". It contains:
            - 'model_state_dict': The state_dict of the model, containing
                current values of learnable parameters,
            - 'optim_state_dict': The state_dict of the optimizer.
                (might be needed for further training)
                Note: This is left out if optimizer=None,
            - 'loss_sum': the current loss, ie. self.stats['loss_sum'].
                (needed for further training).
        The model_state_dict can be loaded with self.load_network_weights().

        If store_metadata is True, then also stores metadata as .json at
        file_name_metadata.json. Included metadata is
            - name of the agent class
            - name of the script invoked to call this function
            - current hash of the git repository the script resides in
            - id of the neptune run used for logging, empty string if not used
            - name of the optimizer class used, "NoneType" if data is stored
                upon quitting the environment.

        WARNING: The function overwrites existing files without warning.

        Args:
            file_name (str): name of the file to be stored. If file extension
                is not specified, default is ".pt". Both ".pt" and ".pth" are
                accepted. The metadata file will append "metadata.json" to
                the provided filename (without extension).
            optimizer (torch.optim.Optimizer): Current optimizer object. If
                None, the optimizer will not be stored. Defaults to None.
            folder (str, optional): target directory for parameters and
                metadata, can be "" to store in the same folder as the script
                calling the function. Defaults to "learned_weights".
            store_metadata (bool, optional): Should metadata be saved?
                Defaults to True.

        Raises:
            ValueError: When user provides a file_name with invalid extension
        """
        name, ext = path.splitext(file_name)
        if ext == '':
            file_name += '.pt'
        elif ext not in ('.pt', '.pth'):
            raise ValueError(
                'Invalid file extension, must be .pt, .pth or not specified.'
                )

        # ensure that folder exists prior to saving (torch needs this)
        if not path.isdir(folder) and folder != '':
            mkdir(folder)

        # save learnable parameters
        if not folder.endswith('/') and folder != '':
            folder += '/'
        full_path = folder + file_name
        if optimizer:
            torch.save(obj={
                'model_state_dict': self.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'loss_sum': self.stats['loss_sum']
                    },
                    f=full_path)
        else:
            torch.save(obj={
                'model_state_dict': self.state_dict(),
                'loss_sum': self.stats['loss_sum']
                    },
                    f=full_path)

        if store_metadata:
            metadata = {
                'agent_class_name': self.__class__.__name__,
                'script_name': path.basename(__file__),
                'current_sha': git.Repo(
                                        search_parent_directories=True
                                        ).head.object.hexsha,
                'neptune_id': self.neptune_id,
                'optim_name': optimizer.__class__.__name__
            }
            meta_filename = folder + name + '_metadata' + '.json'
            with open(meta_filename, 'w') as write_file:
                json.dump(metadata, write_file)

    def load_network_weights(self, filepath: str) -> None:
        """Load network weights (stored with self.save_agents()) to the agent.

        self needs to have the same (learnable) structure as the model whose
        parameters are stored at filepath, but does not need to have anything
        else in common with the original model.

        Args:
            filepath (str): Path to the .pt or .pth file to load from
        """
        self.load_state_dict(torch.load(filepath)['model_state_dict'])


class DirectPolicyAgent_large(DirectPolicyAgent):
    """Modifies DirectPolicyAgent() with a much larger network.
    """
    def __init__(self, **kwargs):
        """Instantiate agent object and layers. See Player() for kwargs.
        """
        DirectPolicyAgent.__init__(self, **kwargs)
        self.L1 = nn.Linear(42, 300)
        self.L2 = nn.Linear(300, 500)
        self.L3 = nn.Linear(500, 1000)
        self.L4 = nn.Linear(1000, 600)
        self.L5 = nn.Linear(600, 200)
        self.L6 = nn.Linear(200, 100)

    def forward(self, x):
        """Pass a game state through the network to decide which move to make.

        Args:
            x (Tensor): Flattened matrix representation of game state.

        Returns:
            Tensor: Probability for each column. The final layer is softmax,
                so the output tensor sums to 1.
        """
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.L3(x)
        x = F.relu(x)
        x = self.L4(x)
        x = F.relu(x)
        x = self.L5(x)
        x = F.relu(x)
        x = self.L6(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)


class DirectPolicyAgent_mini(DirectPolicyAgent):
    """Modifies DirectPolicyAgent() with a much smaller network.
    """
    def __init__(self, **kwargs):
        """Instantiate agent object and layers. See Player() for kwargs.
        """
        DirectPolicyAgent.__init__(self, **kwargs)
        self.L1 = nn.Linear(42, 300)
        self.final = nn.Linear(300, 7)

    def forward(self, x):
        """Pass a game state through the network to decide which move to make.

        Args:
            x (Tensor): Flattened matrix representation of game state.

        Returns:
            Tensor: Probability for each column. The final layer is softmax,
                so the output tensor sums to 1.
        """
        x = self.L1(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)


class HumanPlayer(Player):
    """Extends Player() to let the user play the game using console input."""
    def __init__(self, **kwargs):
        """Create new HumanPlayer object. See Player() docs for kwargs."""
        Player.__init__(self, **kwargs)

    def select_action(self,
                      game: connect_four,
                      illegal_moves_allowed: bool = True) -> int:
        """Ask for user input to choose a column.

        Args:
            game (connect_four): The current connect four game object
            illegal_moves_allowed (bool, optional): bool indicating whether
                or not illegal moves are allowed. Defaults to True.
                This argument is not used by the method, but is included
                since every select_action method needs to have the argument.

        Returns:
            int: The column to place the piece in, 0-indexed.
        """
        # Calculating legal_cols since legal_moves may be an empty list
        chosen_col = int(input("Choose column: ")) - 1
        while chosen_col not in game.legal_cols():
            # 1-indexed
            printable_legals = [col+1 for col in game.legal_cols()]
            print(f"Illegal column. Choose between {printable_legals}.")
            chosen_col = int(input("Choose column: ")) - 1
        return chosen_col


class MinimaxAgent(Player):
    """Extends Player() to use the Minimax-algorithm for choosing moves.
    """
    def __init__(self, max_depth=1, **kwargs):
        """Construct MinimaxAgent() object. See Player() for kwargs.

        Args:
            max_depth (int, optional): Depth of the game tree to analyze, ie.
                number of steps to look forward. Note that the game tree grows
                exponentially and that the implementation isn't very
                efficient. Should not exceed 2 if used when training agents.
                Defaults to 1.
        """
        Player.__init__(self, **kwargs)
        self.max_depth = max_depth

    def select_action(self,
                      game: connect_four,
                      illegal_moves_allowed: bool = True) -> int:
        """Use the Minimax-algorithm to determine the next move to make.

        Args:
            game (connect_four): The current connect four game object.
            illegal_moves_allowed (bool, optional): bool indicating whether
                or not illegal moves are allowed.
                The method will always play legal moves even if
                illegal_moves_allowed=True.
                Defaults to True.

        Returns:
            int: Index of the chosen column.
        """
        best_score = self.params["loss_reward"] - 1
        best_col = None
        possible_ties = []

        # Make sure the MinimaxAgent always have legal_moves to choose from
        # The reason for this is that this agent can not make illegal moves
        for col in game.legal_cols():
            game.place_piece(column=col, piece=self.playerPiece)
            score = self.minimax(game=game, depth=0, maximizing=False)
            game.remove_piece(column=col)
            if score == self.params["not_ended_reward"] and \
                    score >= best_score:
                possible_ties.append(col)
                best_score = score
                best_col = col
            elif score > best_score:
                best_score = score
                best_col = col

        if best_score == self.params["not_ended_reward"]:
            return random.choice(possible_ties)
        return best_col

    def minimax(self,
                game: connect_four,
                depth: int,
                maximizing: bool) -> float:
        """Runs the Minimax algorithm on the board.

        Args:
            game (connect_four): The current connect four game object.
            depth (int): The current game tree depth of the minimax algorithm.
            maximizing (bool): True if it is the maximizing player, and false
                if it is the minimizing player.

        Returns:
            float: The best score obtained before reaching self.max_depth.
        """
        if game.winning_move():
            if not maximizing:
                return self.params["win_reward"]/(depth+1)
            else:
                return self.params["loss_reward"]/(depth+1)
        if game.is_tie():
            return self.params["tie_reward"]
        if depth > self.max_depth:
            return self.params["not_ended_reward"]

        if maximizing:
            best_score = None
            for col in game.legal_cols():
                game.place_piece(column=col, piece=self.playerPiece)
                score = self.minimax(game=game,
                                     depth=depth+1,
                                     maximizing=False)
                game.remove_piece(column=col)
                if best_score is None:
                    best_score = score
                elif score > best_score:
                    best_score = score
            return best_score
        else:
            best_score = None
            for col in game.legal_cols():
                game.place_piece(column=col, piece=self.playerPiece*-1)
                score = self.minimax(game=game,
                                     depth=depth+1,
                                     maximizing=True)
                game.remove_piece(column=col)
                if best_score is None:
                    best_score = score
                elif score < best_score:
                    best_score = score
            return best_score


class TDAgent(DirectPolicyAgent):
    """Agent using TD(Lambda), extends methods of DirectPolicyAgent.

    Inspired by Tesauro's implementation of TD-Gammon as presented in RLbook.

    Overwrites the following methods:

    Adds the following methods:
    """
    def __init__(self, train=True, **kwargs):
        """Construct TDAgent object.

        Does not call the init of DirectPolicyAgent to avoid copying its
        network architecture.
        """
        Player.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.L1 = nn.Linear(84, 50)
        self.L2 = nn.Linear(50, 1)
        self.is_training = train
        # creating eligibility traces
        self.eligibility_dict = {}
        for name, param in self.named_parameters():
            self.eligibility_dict[name] = torch.zeros(param.shape)
        self.gamma = 0.9
        self.Lambda = 1
        self.alpha = 0.1

    def forward(self, x):
        """Pass a game state through the network to estimate its value.

        Args:
            x (Tensor): Flattened binary representation of game state.

        Returns:
            Tensor: Probability for each column. The final layer is softmax,
                so the output tensor sums to 1.
        """
        x = self.L1(x)
        x = F.sigmoid(x)
        x = self.L2(x)
        return F.sigmoid(x)

    def represent_binary(self, game_state):
        """_summary_

        Args:
            game_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # NOTE: alternative approach (requires 2 extra nodes in self.L1)
        # p1_positions = [1 if p == 1 else 0 for p in x]
        # pm1_positions = [1 if p == -1 else 0 for p in x]
        # binary_game_state = p1_positions + pm1_positions
        # # make sure that playerPiece corresponds to current player turn
        # if self.playerPiece == 1:
        #     binary_game_state.append(1)
        #     binary_game_state.append(0)
        # else:
        #     binary_game_state.append(0)
        #     binary_game_state.append(1)

        flattened_board = torch.from_numpy(game_state).float().flatten()
        opponent_positions = [1 if p == self.playerPiece*-1 else 0
                              for p in flattened_board]
        own_positions = [1 if p == self.playerPiece else 0
                         for p in flattened_board]
        binary_game_state = opponent_positions + own_positions
        binary_game_state = torch.FloatTensor(binary_game_state)
        return binary_game_state

    def calculate_rewards(self) -> None:
        # NOTE: Only defining to make sure this is not messed with by others
        pass

    def select_action(self,
                      game: connect_four,
                      illegal_moves_allowed: bool = False):
        """
        Args:
            board (np.ndarray): _description_
            illegal_moves_allowed (list, optional): UNUSED.

        Returns:
            _type_: _description_
        """
        values_dict = {}  # Initialise dictionary of move:v_hat pairs
        legal_moves = game.legal_cols()
        for move in legal_moves:
            game.place_piece(column=move, piece=self.playerPiece)
            next_board = game.return_board()
            binary_rep = self.represent_binary(next_board)
            with torch.no_grad():
                v_hat = self.forward(binary_rep)
            values_dict[move] = v_hat
            game.remove_piece(column=move)

        best_move = max(values_dict, key=values_dict.get)
        if self.is_training:
            self.incremental_update(self,
                                    game=game,
                                    best_move_valuation=values_dict[best_move],
                                    best_move=best_move)
        return best_move

    def update_agent(self, optimizer=None) -> None:
        # NOT USED
        pass

    def incremental_update(self,
                           game: connect_four,
                           best_move_valuation: float,
                           best_move: int) -> None:

        game.place_piece(column=best_move,
                         piece=self.playerPiece)
        reward = self.params["win_reward"] if game.winning_move() else 0
        game.remove_piece(column=best_move)

        # reset gradient
        self
        # update each part of weights
        v_hat = self.forward(x=self.represent_binary(
            game_state=game.return_board)
            )
        v_hat.backward()
        with torch.no_grad():
            for name, param in self.named_parameters:
                # update eligibility trace
                self.eligibility_dict[name] = \
                    self.gamma * self.Lambda * self.eligibility_dict[name]\
                    + param.grad
                # multiply with new evidence
                w_change = self.alpha * \
                    (reward + self.gamma*best_move_valuation - v_hat)\
                    * self.eligibility_dict[name]
                # adding to weights
                param += w_change
