import numpy as np
import pytest
from bayesian import StateGenerator, sample_observation, belief_update, belief_predict, initialize_belief


class TestBayesianInference:

    @pytest.mark.parametrize("initial_state, observation_list, prior_style", [
        (
                ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
                [(3, 4)], "uniform",
        ),
        (
                ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
                [(3, 4), (3, 4)], "uniform",
        ),
        (
                ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
                [(3, 4), (3, 5)], "dirac",
        ),
    ])
    def test_example_observations(self, initial_state, observation_list, prior_style):
        """
        Test the observation model and Bayesian updates.
        """
        positions, dimensions = initial_state
        belief = initialize_belief(initial_state, style=prior_style)
        assert belief.shape == dimensions, "Belief should match board dimensions."

        for observation in observation_list:
            posterior = belief_update(belief, observation, initial_state)
            assert posterior.shape == dimensions, "Posterior should match board dimensions."
            assert np.all(posterior >= 0), "Posterior probabilities must be non-negative."
            assert np.isclose(np.sum(posterior), 1), "Posterior should sum to 1."

    @pytest.mark.parametrize("initial_state, action_list, prior_style", [
        (
                ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
                [(0, 0)], "uniform",
        ),
        (
                ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
                [(0, 0), (0, 1)], "uniform",
        ),
        (
                ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
                [(0, 0), (1, 0)], "dirac",
        ),
    ])
    def test_example_actions(self, initial_state, action_list, prior_style):
        """
        Test the action model and prediction updates.
        """
        positions, dimensions = initial_state
        belief = initialize_belief(initial_state, style=prior_style)
        assert belief.shape == dimensions, "Belief should match board dimensions."

        for action in action_list:
            posterior = belief_predict(belief, action, initial_state)
            assert posterior.shape == dimensions, "Posterior should match board dimensions."
            assert np.all(posterior >= 0), "Posterior probabilities must be non-negative."
            assert np.isclose(np.sum(posterior), 1), "Posterior should sum to 1."
