import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATASET_DIR = "/home/miranjo/phasic_tonic/data/raw/OSbasic/"
OUTPUT_DIR  = "/home/miranjo/phasic_tonic/data/processed/OSbasic/"

def rem_extract(lfp, sleep_trans):
    """
    Extract REM sleep data from a LFP using sleep transition times.

    Parameters:
        lfp (numpy.ndarray): A NumPy array.
        sleep_trans (numpy.ndarray): A NumPy array containing pairs of sleep transition times.

    Returns:
        list of numpy.ndarray: A list of NumPy arrays, each representing a segment of REM sleep data.
    """
    rems = []

    for rem in sleep_trans:
        t1 = int(rem[0])
        t2 = int(rem[1])
        rems.append(lfp[t1:t2])

    return rems

def get_rem_states(states, sample_rate):
    """
    Extract consecutive REM (Rapid Eye Movement) sleep states and their start
    and end times from an array of sleep states.

    Parameters:
    - states (numpy.ndarray): One-dimensional array of sleep states.
    - sample_rate (int): The sample rate of the data.

    Returns:
    numpy.ndarray: An array containing start and end times of consecutive REM
    sleep states. Each row represents a pair of start and end times.

    Note:
    - Sleep states are represented numerically. In this function, REM sleep
      states are identified by the value 5 in the 'states' array.

    Example:
    ```python
    import numpy as np

    # Example usage:
    sleep_states = np.array([1, 2, 5, 5, 5, 3, 2, 5, 5, 4, 1])
    sample_rate = 2500  # Example sample rate in Hz
    rem_states_times = get_rem_states(sleep_states, sample_rate)
    print(rem_states_times)
    ```
    """
    try:
        # Ensure the sleep states array is one-dimensional.
        states = np.squeeze(states)
        # Find the indices where the sleep state is equal to 5, indicating REM sleep.
        rem_state_indices = np.where(states == 5)[0]
        
        # Check if there are no REM states. If so, return an empty array.
        if len(rem_state_indices) == 0:
            return np.array([])
        # Calculate the changes between consecutive REM state indices.
        rem_state_changes = np.diff(rem_state_indices)
        # Find the indices where consecutive REM states are not adjacent.
        split_indices = np.where(rem_state_changes != 1)[0] + 1
        # Add indices to split consecutive REM states, including the start and end indices.
        split_indices = np.concatenate(([0], split_indices, [len(rem_state_indices)]))
        # Create an empty array to store start and end times of consecutive REM states.
        consecutive_rem_states = np.empty((len(split_indices) - 1, 2))
        # Iterate through the split indices to extract start and end times.
        for i, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
            start = rem_state_indices[start] * int(sample_rate)
            end = rem_state_indices[end - 1] * int(sample_rate)
            consecutive_rem_states[i] = np.array([start, end])
        # Convert the array to a numpy array.
        ##consecutive_rem_states = np.array(consecutive_rem_states)
        # Create a mask to filter out consecutive REM states with negative duration.
        null_states_mask = np.squeeze(np.diff(consecutive_rem_states) > 0)
        consecutive_rem_states = consecutive_rem_states[null_states_mask]
        # Return the array containing start and end times of consecutive REM states.
        return consecutive_rem_states
    # Handle the case where an IndexError occurs, typically due to an empty array.
    except IndexError as e:
        print(f"An IndexError occurred in get_rem_states: {e}")
        return np.array([])  # or any default value you want
    
def create_name(fname):
    print(fname)

    # Extract the rat number
    match = re.search(r"(\d+)/", fname)
    rat_num = match.group(1) if match else None
    title_name = 'Rat' + rat_num
    
    # Extract the condition
    if ("OR_N" in fname) or ("OR-N" in fname):
        title_name += '_' + "OR-N"
    elif ("OD_N" in fname) or ("OD-N" in fname):
        title_name += '_' + "OD-N"
    else:
        match = re.search(r"(?!OS|SD)[A-Z]{2}", fname)
        condition = match.group(0) if match else None
        title_name += '_' + condition

    # Extract the post trial number
    match = re.search(r"post_trial(\d+)", fname, flags=re.IGNORECASE)
    posttrial_num = match.group(1) if match else None
    title_name += '_4_' + 'posttrial' + str(posttrial_num)
    print("Filename: ", fname, "\nExtracted:", title_name)

    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number

    return title_name

pattern1 = r".*post_trial.*"
mapped = {}

for root, dirs, fils in os.walk(DATASET_DIR):
    for dir in dirs:
        dir = (os.path.join(root, dir))
        # Check if the directory is a post trial directory
        if re.match(pattern1, dir, flags=re.IGNORECASE):
            dir = Path(dir)
            print("MATCH: ", dir)
            HPC_file = next(dir.glob("*HPC*"))
            states = next(dir.glob('*states*'))
            mapped[states] = HPC_file
        else:
            print("No match:", dir)

for state in mapped.keys():
    hpc = mapped[state]
    lfp = loadmat(hpc)['HPC']
    lfp = np.squeeze(lfp)
    sleep = loadmat(state)
    states = np.squeeze(sleep['states'])

    if(np.any(states == 5)):
        rem_transitions = get_rem_states(states, 1000).astype(int)
        if rem_transitions.ndim == 3:
            rem_transitions = np.squeeze(rem_transitions, 0)

        lfpREM = rem_extract(lfp, rem_transitions)
        title = create_name(str(hpc.relative_to(hpc.parent.parent.parent.parent))) 
        fname = OUTPUT_DIR + title
        np.savez(fname, *lfpREM)