import numpy as np

class StateGenerator:

    def __init__(self, nrows=8, ncols=7, npieces=10):
        """
        Initialize a generator for sampling valid states from
        an npieces dimensional state space.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.npieces = npieces
        self.rng = np.random.default_rng()

    def sample_state(self):
        """
        Samples a self.npieces length tuple.

        Output:
            Returns a state. A state is as 2-tuple (positions, dimensions), where
             -  Positions is represented as a list of position (c,r) tuples 
             -  Dimensions is a 2-tuple (self.nrows, self.ncols)

            For example, if the dimensions of the board are 2 rows, 3 columns, and the number of pieces
            is 4, then a valid return state would be ([(0, 0) , (1, 0), (2, 0), (1, 1)], (2,3))
        """
        ## Returns positions in decoded format. i.e. list of (c,r) i.e. (x,y)
        ## Without loss of generalization, we assume that positions[1:] are fixes; only
        ## positions[0] will be moved
        positions = self.rng.choice(self.nrows * self.ncols, size=self.npieces, replace=False)
        pos = list(self.decode(p) for p in positions)
        return pos, (self.nrows, self.ncols)

    def decode(self, position):
        r = position // self.ncols
        c = position - self.ncols * r
        return (c, r)


def sample_observation(state):
    """
        Given a state, sample an observation from it. Specifically, the positions[1:] locations are
        all known, while positions[0] should have a noisy observation applied.

        Input:
            State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

        Returns:
            A tuple (position, distribution) where:
             - Position is a sampled position which is a 2-tuple (c, r), which represents the sampled observation
             - Distribution is a 2D numpy array representing the observation distribution

        NOTE: the array representing the distribution should have a shape of (nrows, ncols)
    """
    # # We need to return the first element in pos which is the first part of the state
    # pos, dimensions = state
    #
    # # First element in pos
    # movable_piece = pos[0]
    #
    # # We now need to get the distribution, this will be a nrows x ncols grid.
    # # Need to also check if there were any pieces above, below, and to the right and left of that piece
    # probability = .6
    # count = 0
    # above = right = below = left = -1
    #
    # # Create a numpy array to return
    # distribution = np.zeros(dimensions)
    #
    # if movable_piece[1] < dimensions[0]:
    #     # Checking for above
    #     for i in range(1, len(pos)):
    #         # Row value needs to be greater by 1 and column needs to be the same
    #         if movable_piece[1] + 1 == pos[i][1] and movable_piece[0] == pos[i][0]:
    #             count = count + 1
    #             # Get the pos[i] column and row value then update that column and row value in numpy array to 10
    #             above = .1
    #             distribution[pos[i][1]][pos[i][0]] = above
    #             # In case more than one piece is at the same spot (should not happen)
    #             break
    #
    # if movable_piece[0] < dimensions[1]:
    #     # Checking for right
    #     for i in range(1, len(pos)):
    #         # Row value needs to be the same, column needs to be larger for the pos by 1
    #         if movable_piece[1] == pos[i][1] and movable_piece[0] + 1 == pos[i][0]:
    #             count = count + 1
    #             right = .1
    #             distribution[pos[i][1]][pos[i][0]] = right
    #             # In case more than one piece is at the same spot (should not happen)
    #             break
    #
    # if movable_piece[1] > 0:
    #     # Checking for the bottom
    #     for i in range(1, len(pos)):
    #         if movable_piece[1] - 1 == pos[i][1] and movable_piece[0] == pos[i][0]:
    #             count = count + 1
    #             bottom = .1
    #             distribution[pos[i][1]][pos[i][0]] = bottom
    #             # In case more than one piece is at the same spot (should not happen)
    #             break
    #
    # if movable_piece[0] > 0:
    #     # Checking for the left
    #     for i in range(1, len(pos)):
    #         if movable_piece[0] - 1 == pos[i][0] and movable_piece[1] == pos[i][0]:
    #             count = count + 1
    #             left = .1
    #             distribution[pos[i][1]][pos[i][0]] = left
    #             # In case more than one piece is at the same spot (should not happen)
    #             break
    #
    # # Calculate total probability piece is where we believe it is
    # probability += count * 0.1
    #
    # # Adding it to our distribution
    # distribution[movable_piece[1]][movable_piece[0]] = probability
    #
    # return movable_piece, distribution


    positions, dimensions = state
    movable_piece = positions[0]  # The piece to track
    nrows, ncols = dimensions
    distribution = np.zeros(dimensions)
    adjacent_prob = 0.1
    prob = 0.6
    # Define movement directions (delta for column, delta for row)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    c, r = movable_piece
    distribution[r, c] += prob

    # Iterate over all directions to check neighbors
    for delta_col, delta_row in directions:
        neighbor_col = c + delta_col
        neighbor_row = r + delta_row
        # Check if the neighbor is within bounds and not occupied
        if (neighbor_col >= 0 and neighbor_col < ncols and neighbor_row >= 0 and neighbor_row < nrows
                and (neighbor_col, neighbor_row) not in positions[1:]):
            distribution[neighbor_row, neighbor_col] = adjacent_prob
        else:
            distribution[r, c] += adjacent_prob

    # Normalization step
    distribution /= np.sum(distribution)
    flattened_distribution = distribution.flatten()
    sampled_index = np.random.choice(ncols * nrows, p=flattened_distribution)
    sampled_column = sampled_index % ncols
    sampled_row = sampled_index // ncols

    return (sampled_row, sampled_column), distribution

def sample_transition(state, action):
    """
    Given a state and an action, 
    returns:
         a resulting state, and a probability distribution represented by a 2D numpy array
    If a transition is invalid, returns None for the state, and a zero probability distribution
    NOTE: the array representing the distribution should have a shape of (nrows, ncols)

    Inputs:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        Action: a 2-tuple (dc, dr) representing the difference in positions of position[0] as a result of
                executing this transition.

    Outputs:
        A 2-tuple (new_position, transition_probabilities), where
            - new_position is:
                A 2-tuple (new_column, new_row) if the action is valid.
                None if the action is invalid.
            - transition_probabilities is a 2D numpy array with shape (nrows, ncols) that accurately reflects
                the probability of ending up at a certain position on the board given the action. 
    """
    # Need to check if the addition of the action takes it outside of the board
    pos, dimensions = state
    movable_piece = pos[0]
    new_location = (movable_piece[0] + action[0], movable_piece[1] + action[1])

    # Initialize transition_probabilities
    transition_probabilities = np.zeros(dimensions)
    # Need a converted new_location value for row-major order
    Row_major_new_location = (dimensions[0] - new_location[1], new_location[0])
    # Now to check if a valid location
    if (Row_major_new_location[0] > dimensions[1] or Row_major_new_location[0] < 0
            or Row_major_new_location[1] > dimensions[0] or Row_major_new_location[1] < 0)\
            or new_location in pos[1:]:
        return (None, transition_probabilities) # All probabilities should be 0 since an invalid action
    else:
        transition_probabilities[Row_major_new_location[0]][Row_major_new_location[1]] = 100
        return (new_location, transition_probabilities)


def initialize_belief(initial_state, style="uniform"):
    """
    Create an initial belief, based on the type of belief we want to start with

    Inputs:
        Initial_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        style: an element of the set {"uniform", "dirac"}

    Returns:
        an initial belief, represented by a 2D numpy array with shape (nrows, ncols)

    NOTE:
        The array representing the distribution should have a shape of (nrows, ncols).
        The occupied spaces (if any) should be zeroed out in the belief.
        We define two types of priors: a uniform prior (equal probability over all
        unoccupied spaces), and a dirac prior (which concentrates all the probability
        onto the actual position on the piece).

    """
    pos, dimensions = initial_state
    initial_belief = np.zeros(dimensions)
    nrows, ncols = dimensions
    if style == "uniform":

        occupied_spaces_row_major = []
        for i in range(1, len(pos)):
            occupied_spaces_row_major.append((pos[i][1], pos[i][0]))

        total_locations = dimensions[0] * dimensions[1]

        # We are getting the total amount of spots subtracted by amount of occupied spots
        unoccupied_spaces = total_locations - (len(pos) -1) # We can consider pos[0] to be unoccupied
        probability_uniform = 1 / unoccupied_spaces

        # Nested for loop
        for row in range(0, nrows):
            for col in range(0, ncols):
                if (row, col) not in occupied_spaces_row_major:
                    initial_belief[row, col] = probability_uniform

    if style == "dirac":
        actual_pos = pos[0]
        row_major_actual_pos = (actual_pos[1], actual_pos[0])
        initial_belief[row_major_actual_pos[0]][row_major_actual_pos[1]] = 1

    return initial_belief


def belief_update(prior, observation, reference_state):
    """
    Given a prior an observation, compute the posterior belief

    Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        observation: a 2-tuple (col, row) representing the observation of a piece at a position
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    # pos, dimensions = reference_state
    # nrow, ncol = dimensions
    # c_observation, r_observation = observation
    # _, observation_dist = sample_observation((pos, dimensions))
    #
    # pos[0] = (c_observation, r_observation)
    # posterior = np.zeros(dimensions)
    #
    # posterior = observation_dist * prior
    # # Normalization of the posterior distribution
    # posterior /= np.sum(posterior)
    #
    # return posterior
    pos, dimensions = reference_state
    nrow, ncol = dimensions
    c_observation, r_observation = observation
    temp_state = ([(c_observation, r_observation)] + pos[1:], dimensions)
    _, observation_dist = sample_observation(temp_state)

    # pos[0] = (c_observation, r_observation)

    posterior = observation_dist * prior

    for p in pos[1:]:
        row, col = p[1], p[0]
        posterior[row, col] = 0

    # Normalization.
    total = np.sum(posterior)
    if total > 0:
        posterior /= total

    return posterior

def belief_predict(prior, action, reference_state):
    """
    Given a prior, and an action, compute the posterior belief.

    Actions will be given in terms of dc, dr

   Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        action: a 2-tuple (dc, dr) as defined for action in sample_transition
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    pos, dimensions = reference_state
    nrow, ncol = dimensions
    dc, dr = action
    posterior = np.zeros(dimensions)

    for row in range(nrow):
        for col in range(ncol):
            new_col, new_row = col + dc, row + dr
            if (new_col >= 0 and new_col < ncol and new_row >= 0 and new_row < nrow):
                posterior[new_row, new_col] += prior[row, col]
                if (new_col, new_row) in pos[1:]:
                    posterior[new_row, new_col] = 0
    posterior /= np.sum(posterior)

    return posterior

if __name__ == "__main__":
    gen = StateGenerator()
    initial_state = gen.sample_state()
    obs, dist = sample_observation(initial_state)
    print(initial_state)
    print(obs)
    print(dist)
    b = initialize_belief(initial_state, style="uniform")
    print(b)
    b = belief_update(b, obs, initial_state)
    print(b)
    b = belief_predict(b, (1, 0), initial_state)
    print(b)
