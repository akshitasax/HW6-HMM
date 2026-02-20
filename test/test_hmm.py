import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    # Load toy HMM parameters and test sequences
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    # Unpack HMM parameters
    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    # Unpack sequences
    test_obs_seq = mini_input['observation_state_sequence']
    viterbi_expected = mini_input['best_hidden_state_sequence']

    # Create HMM object
    mini_hmm_obj = HiddenMarkovModel(
        observation_states=observation_states,
        hidden_states=hidden_states,
        prior_p=prior_p,
        transition_p=transition_p,
        emission_p=emission_p
    )

    # Test Forward algorithm
    forward_prob = mini_hmm_obj.forward(test_obs_seq)
    assert forward_prob > 0.0 and forward_prob <= 1.0

    # Test Viterbi algorithm
    viterbi_path = mini_hmm_obj.viterbi(test_obs_seq)
    # Check if output is correct length and valid state indices
    assert len(viterbi_path) == len(test_obs_seq)
    # viterbi_path contains hidden state labels, so check membership in hidden_states
    assert all(state in hidden_states for state in viterbi_path)

    # ground-truth path is provided, compare against that
    if viterbi_expected is not None:
        assert np.array_equal(viterbi_path, viterbi_expected)

    # Edge case 1: Empty sequence should not crash, and return probability 1 or 0 correctly
    empty_seq = np.array([], dtype=test_obs_seq.dtype)
    try:
        forward_empty = mini_hmm_obj.forward(empty_seq)
        assert forward_empty == 1.0 or forward_empty == 0.0
        viterbi_empty = mini_hmm_obj.viterbi(empty_seq)
        assert len(viterbi_empty) == 0
    except Exception:
        pytest.fail("HMM implementation does not handle empty sequences.")

    # Edge case 2: Sequence of an observation never seen (invalid state)
    invalid_seq = np.array(['INVALID_OBS'], dtype=test_obs_seq.dtype)
    with pytest.raises(KeyError):
        mini_hmm_obj.forward(invalid_seq)
    with pytest.raises(KeyError):
        mini_hmm_obj.viterbi(invalid_seq)



def test_full_weather():

    # Load the HMM parameters and expected outputs for the full weather problem
    weather_hmm = np.load("./data/full_weather_hmm.npz", allow_pickle=True)
    weather_io = np.load("./data/full_weather_sequences.npz", allow_pickle=True)

    # Unpack the HMM parameters
    observation_states = weather_hmm['observation_states']
    hidden_states = weather_hmm['hidden_states']
    prior_p = weather_hmm['prior_p']
    transition_p = weather_hmm['transition_p']
    emission_p = weather_hmm['emission_p']

    # Unpack sequences and expected outputs
    test_obs_seq = weather_io['observation_state_sequence']
    viterbi_expected = weather_io['best_hidden_state_sequence']

    # Create HMM object
    hmm_obj = HiddenMarkovModel(
        observation_states=observation_states,
        hidden_states=hidden_states,
        prior_p=prior_p,
        transition_p=transition_p,
        emission_p=emission_p
    )

    # Test Forward algorithm
    forward_prob = hmm_obj.forward(test_obs_seq)
    assert forward_prob > 0.0 and forward_prob <= 1.0

    # Test Viterbi algorithm
    viterbi_path = hmm_obj.viterbi(test_obs_seq)
    # Check that output is right length, and only valid states
    assert len(viterbi_path) == len(test_obs_seq)
    assert all(state in hidden_states for state in viterbi_path)
    # Check that it matches provided "best" path
    if viterbi_expected is not None:
        assert np.array_equal(viterbi_path, viterbi_expected)














