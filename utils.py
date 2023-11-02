import matplotlib.pyplot as plt
import numpy as np

def plot(num_hash_tables: np.ndarray, rh_times: np.ndarray, qa_times: np.ndarray, title: str, labels: list, output_path: str = './output.png') -> None:
    """
    Function to plot the results of the experiments.

    Parameters:
        - num_hash_tables (np.ndarray): array with the number of hash tables.
        - rh_times (list): list of the average time of RH.
        - qa_times (list): list of the average time of QA.
        - title (str): title of the plot.
        - labels (list): array with the labels of the plots.
        - output_path (str, optional): path to save the plot (default './output.png').
    """
    # Convert to numpy array if necessary.
    if isinstance(num_hash_tables, list):
        num_hash_tables = np.array(num_hash_tables)
    
    fig, ax = plt.subplots()

    # Plot the data and add a legend.
    ax.plot(num_hash_tables.astype('str'), rh_times, 'o-', color='navy', label="RH")
    ax.plot(num_hash_tables.astype('str'), qa_times, 'o-', color='orangered', label="QA")
    ax.legend()

    # Add a table.
    table = plt.table(cellText=[rh_times, qa_times], rowLabels=['Random Hyperplane', 'Query Aware'], colLabels=num_hash_tables, loc='bottom', cellLoc='center', bbox=[0, -0.4, 1, 0.2])

    # Add title and axis names.
    plt.suptitle(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    # Save the plot.
    plt.savefig(output_path, bbox_inches='tight', dpi=300)