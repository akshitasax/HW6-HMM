import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states

        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))} #order or observations

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))} #order of hidden states

        self.prior_p = prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        sequence = input_observation_states

        if len(sequence) == 0:
            return 0
    
        # Step 2. Calculate probabilities

        p_val = None

        for i in range(len(sequence)):
            obs_idx = self.observation_states_dict[sequence[i]]
            if i == 0:
                # Find most likely starting hidden state
                obs_idx = self.observation_states_dict[sequence[i]]
                p_val = self.prior_p * self.emission_p[:, obs_idx]
                # initial prob of first obs given any state
            else:
                # Find most likely next hidden state from previous state
                obs_idx = self.observation_states_dict[sequence[i]]
                p_val = (p_val @ self.transition_p) * self.emission_p[:, obs_idx] # prob of seeing obs given any state * prob of shifting to that state from any prev state
        
        final_p = np.sum(p_val)

        # Step 3. Return final probability 
        return final_p


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        sequence = decode_observation_states
        T = len(sequence)
        N = len(self.hidden_states)

        # Handle empty sequence case
        if T == 0:
            return []

        viterbi_table = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)

        obs_idx = self.observation_states_dict[sequence[0]]
        for s in range(N): # for each possible hidden state:
            # prob of being in that hidden state * prob of seeing that obs emitted from that hidden state
            viterbi_table[0, s] = self.prior_p[s] * self.emission_p[s, obs_idx]
            # at t=0, most likely state = 0
            backpointer[0, s] = 0  # Doesn't matter for t=0

        # at initialization, row 0 (time point 0), has been filled with the prob of seeing a given obs at each possible state

        # Step 2. Calculate Probabilities
        for t in range(1, T):  # at each time point (note: should go through last time index T-1)
            obs_idx = self.observation_states_dict[sequence[t]]
            for s in range(N):
                trans_probs = viterbi_table[t-1] * self.transition_p[:, s]
                max_prev_state = np.argmax(trans_probs)
                viterbi_table[t, s] = trans_probs[max_prev_state] * self.emission_p[s, obs_idx]
                backpointer[t, s] = max_prev_state  # store the index, not mapped id

        # 3. Path Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_table[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        # Convert best_path from indices to hidden state labels
        best_path_labels = [self.hidden_states[idx] for idx in best_path]
        return best_path_labels
        