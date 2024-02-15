import sys
import os

import json
import numpy as np
import time
from tqdm import tqdm

from utils import *

class RandomHyperplaneLSH:
    def __init__(self, num_hash_tables: int, num_hash_functions: int, seed: int = 0):
        '''
        Initialize the RandomHyperplaneLSH object with the parameters needed for the Local Sensitive Hashing (LSH) algorithm.

        Parameters:
        - num_hash_tables (int): represents the number of hash tables (M) used in the LSH algorithm.
        Hash tables are used to create multiple projections of the data and facilitate the search for similar neighbors.
        
        - num_hash_functions (int): represents the number of hash functions (k) used to project the input data into a lower-dimensional space.
        These hash functions generate hash values that are used to perform approximate search for similar neighbors.

        - seed (int, optional): seed value for the random number generator (default 0).
        If provided, it allows the same results to be reproduced in subsequent runs of the algorithm.
        If not provided, the random number generator will use a different random seed on each run.
        '''
        # Define number of hash table (M).
        self.num_hash_tables: int = num_hash_tables
        # Define number of hash functions (k).
        self.num_hash_functions: int = num_hash_functions

        # Set seed for reproducibility.
        np.random.seed(seed)
        
    def _hash_value(self, input_vector: np.ndarray, mapping_vector: np.ndarray) -> np.ndarray:
        '''
        Function to calculate the binary hash value for the input vector.

        Parameters:
            - input_vector (np.ndarray): input vector for which to calculate the hash.
            - mapping_vector (np.ndarray): mapping vector used to reduce the dimensionality of the input vector.

        Returns:
            - np.ndarray: binary vector where each element is 1 if the corresponding hash value is positive, and 0 otherwise.
        '''
        # Calculate matrix product between input vector and random vector.
        hash_vector: np.ndarray = np.dot(input_vector, mapping_vector)

        # Return binary vector based on sign of hash vector's values (1 if positive, 0 otherwise).
        return (hash_vector >= 0)
    
    def lsh(self, data: np.ndarray, query: np.ndarray, top_n: int) -> tuple[np.ndarray, float, float]:
        '''
        Locality Sensitive Hashing (LSH) algorithm for approximate nearest neighbor search.

        Parameters:
            - data (np.ndarray): array containing the embeddings of data samples to be hashed.
            - query (np.ndarray): array containing embeddings of query for which to find the nearest neighbors samples.
            - top_n (int): number of nearest neighbors to retrieve as recommendations.

        Returns:
            - tuple containing three elements:
                - recommended_samples_index (np.ndarray): array of indices representing the top N similar samples as recommendation items.
                - creation_time (float): time in seconds to create hash tables.
                - recommendation_time (float): time in seconds to retrieve top N similar samples.
        '''
        # Define hash tables to store matching between buckets.
        hash_tables: np.ndarray = np.zeros(shape=(self.num_hash_tables, data.shape[0]), dtype=np.bool_)

        # Start creation time.
        start_creation_time = time.time()

        for i in tqdm(range(self.num_hash_tables), desc='RH Hashing', smoothing=0.05, disable=True, dynamic_ncols=True):
            # Define a random vector R (with size of 768*k) from normal distribution to reduce dimensions of input vector.
            random_vector: np.ndarray = np.random.normal(size=(768, self.num_hash_functions))

            # Calculate binary hash vector for data.
            data_hash: np.ndarray = self._hash_value(data, random_vector)
    
            # Assigned sample to bucket based on his binary hash values.
            data_bucket: np.ndarray = np.sum(2**np.arange(self.num_hash_functions) * data_hash, axis=1, dtype=np.int32)

            # Calculate binary hash vector for query.
            query_hash: np.ndarray = self._hash_value(query, random_vector)

            # Assigned query to bucket based on his binary hash values.
            query_bucket: np.ndarray = np.sum(2**np.arange(self.num_hash_functions) * query_hash, axis=0, dtype=np.int32)

            # Match whether the data bucket is the same as the query bucket (get True if match, False otherwise).
            hash_tables[i] = (data_bucket == query_bucket)

        # Calculate creation time.
        end_creation_time = time.time()
        creation_time = end_creation_time - start_creation_time
        print(f'LSH creation time (s): {creation_time}')

        # Start recommendation time.
        start_recommendation_time = time.time()
        
        # Checks whether a sample makes hashes in the same query bucket in at least one hash table
        # (get True only if sample bucket match query bucket in at least one hash table).
        matching_buckets: np.ndarray = np.any(hash_tables, axis=0)
        
        # Retrieve index of matching bucket.
        matching_buckets_index: np.ndarray = np.where(matching_buckets)[0]

        # Calculate cosine similarity between the matched samples and query.
        cosine_similarity: np.ndarray = np.dot(data[matching_buckets_index], query) / (np.linalg.norm(data[matching_buckets_index], axis=1) * np.linalg.norm(query))

        # Get first N samples sorted according to cosine similarity.
        sorted_indices = np.argsort(cosine_similarity, axis=0)
        recommended_samples_index = matching_buckets_index[sorted_indices][:top_n]

        # Calculate recommendation time.
        end_recommendation_time = time.time()
        recommedation_time = end_recommendation_time - start_recommendation_time
        print(f'LSH recommendation time (s): {recommedation_time}')

        # Return top N similar samples as recommendation items.
        return (recommended_samples_index, creation_time, recommedation_time)

class QueryAwareLSH:
    def __init__(self, num_hash_tables: int, num_hash_functions: int, threshold: int = 2, seed: int = 0):
        '''
        Initialize the QueryAwareLSH object with the parameters needed for the Local Sensitive Hashing (LSH) algorithm.

        Parameters:
        - num_hash_tables (int): represents the number of hash tables (M) used in the LSH algorithm.
        Hash tables are used to create multiple projections of the data and facilitate the search for similar neighbors.
        
        - num_hash_functions (int): represents the number of hash functions (k) used to project the input data into a lower-dimensional space.
        These hash functions generate hash values that are used to perform approximate search for similar neighbors.

        - threshold (int, optional): represents number of times that the Euclidean distance between a hash vector of a data sample and the query’s hash vector is less than or equal to w/2.

        - seed (int, optional): seed value for the random number generator (default 0).
        If provided, it allows the same results to be reproduced in subsequent runs of the algorithm.
        If not provided, the random number generator will use a different random seed on each run.
        '''
        # Define number of hash table (M).
        self.num_hash_tables: int = num_hash_tables
        # Define number of hash functions (k).
        self.num_hash_functions: int = num_hash_functions
        # Define threshold occurences (l).
        self.threshold: int = threshold

        # Set seed for reproducibility.
        np.random.seed(seed)

    def _hash_value(self, input_vector: np.ndarray, mapping_vector: np.ndarray) -> np.ndarray:
        '''
        Function to calculate the binary hash value for the input vector.

        Parameters:
            - input_vector (np.ndarray): input vector for which to calculate the hash.
            - mapping_vector (np.ndarray): mapping vector used to reduce the dimensionality of the input vector.

        Returns:
            - np.ndarray: binary vector where each element is 1 if the corresponding hash value is positive, and 0 otherwise.
        '''
        # Calculate matrix product between input vector and random vector.
        return np.dot(input_vector, mapping_vector)

    def lsh(self, data: np.ndarray, query: np.ndarray, top_n: int, bucket_width: float = 4) -> tuple[np.ndarray, float, float]:
        '''
        Locality Sensitive Hashing (LSH) algorithm for approximate nearest neighbor search.

        Parameters:
            - data (np.ndarray): array containing the data samples to be hashed.
            - query (np.ndarray): query sample for which to find the approximate nearest neighbors.
            - top_n (int): number of nearest neighbors to retrieve as recommendations.
            - bucket_width (float): width of the bucket used in LSH algorithm. Larger values of bucket width may lead to more approximate matches.

        Returns:
            - tuple containing three elements:
                - recommended_samples_index (np.ndarray): array of indices representing the top N similar samples as recommendation items.
                - creation_time (float): time in seconds to create hash tables.
                - recommendation_time (float): time in seconds to retrieve top N similar samples.
        '''
        # Define occurrences tables to store number of times a sample is hashed near the query.
        occurrences: np.ndarray = np.zeros(shape=(data.shape[0]), dtype=np.int32)
        
        # Start creation time.
        start_creation_time = time.time()

        for _ in tqdm(range(self.num_hash_tables), desc='QA Hashing', smoothing=0.05, disable=True, dynamic_ncols=True):
            # Define a random vector R (with size of 768*k) from normal distribution to reduce dimensions of input vector.
            random_vector: np.ndarray = np.random.normal(size=(768, self.num_hash_functions))

            # Calculate hash vector for data.
            data_hash: np.ndarray = self._hash_value(data, random_vector)

            # Calculate hash vector for query.
            query_hash: np.ndarray = self._hash_value(query, random_vector)

            # Calculate Eucldean distance between each data sample and query.
            distances = np.linalg.norm(data_hash - query_hash, axis=1)
            
            # Increment occurrence number of each samples with Euclidian distance from query less then w/2.
            occurrences += (distances <= bucket_width/2)

        # Calculate creation time.
        end_creation_time = time.time()
        creation_time = end_creation_time - start_creation_time
        print(f'LSH creation time (s): {creation_time}')

        # Start recommendation time.
        start_recommendation_time = time.time()

        # Find candidate samples with occurrence higher than threshold.
        candidates_index: np.ndarray = np.where(occurrences >= self.threshold)[0]

        # Calculate similarity between candidate samples and query.
        distances_similarity: np.ndarray = np.linalg.norm(data[candidates_index] - query, axis=1)

        # Get first N samples sorted according to Euclidean distance.
        ranked_candidates: np.ndarray = np.argsort(distances_similarity)
        recommended_samples_index: np.ndarray = candidates_index[ranked_candidates][:top_n]

        # Calculate recommendation time.
        end_recommendation_time = time.time()
        recommendation_time = end_recommendation_time - start_recommendation_time
        print(f'LSH recommendation time (s): {recommendation_time}')

        # Return top N similar samples as recommendation items.
        return (recommended_samples_index, creation_time, recommendation_time)


def load_text(code_examples_path: str, queries_path: str) -> tuple:
    '''
    Function to load code examples, Natural Language queries, and API queries from files.
    
    Parameters:
        - code_examples_path (str): file path of the JSON file containing code examples.
        - queries_path (str): file path of the JSON file containing queries.

    Returns:
        - tuple: a tuple containing three lists:
            - code_examples (list): list of code examples.
            - nl_queries (list): list of Natural Language queries.
            - api_queries (list): list of API queries.
    '''
    # Load code examples.
    with open(code_examples_path, 'r', encoding='utf-8') as f:
        code_examples = json.load(f)

    # Load queries.
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

        # Split each query into Natural Language and API query.
        nl_queries = queries['natural_language']
        api_queries = queries['api']
    
    # Return a tuple containing code examples, natural language queries and API queries.
    return (code_examples, nl_queries, api_queries)

def load_embeddings(code_examples_embeddings_path: str, nl_queries_embeddings_path: str, api_queries_embeddings_path: str) -> tuple:
    '''
    Function to load embeddings for code examples, Natural Language queries, and API queries from files.

    Parameters:
        - code_examples_embeddings_path (str): file path of the pickle file (.pkl) containing embeddings for code examples.
        - nl_queries_embeddings_path (str): file path of the pickle file (.pkl) containing embeddings for Natural Language queries.
        - api_queries_embeddings_path (str): file path of the pickle file (.pkl) containing embeddings for API queries.

    Returns:
        - tuple: a tuple containing three numpy arrays:
            - data_vectors (np.ndarray): numpy array of embeddings for code examples.
            - nl_query_vectors (np.ndarray): numpy array of embeddings for Natural Language queries.
            - api_query_vectors (np.ndarray): numpy array of embeddings for API queries.
    '''
    # Load embeddings for code examples.
    data_vectors = np.load(code_examples_embeddings_path, allow_pickle=True)
    data_vectors = np.array(data_vectors)

    # Load embeddings for Natural Language queries.
    nl_query_vectors = np.load(nl_queries_embeddings_path, allow_pickle=True)
    nl_query_vectors = np.array(nl_query_vectors)

    # Load embeddings for API queries.
    api_query_vectors = np.load(api_queries_embeddings_path, allow_pickle=True)
    api_query_vectors = np.array(api_query_vectors)

    # Return a tuple containing numpy arrays of code examples embeddings, Natural Language queries embeddings and API queries embeddings.
    return (data_vectors, nl_query_vectors, api_query_vectors)


def execute_lsh(output_path: str, algorithm_type: str, queries_type: str, code_examples_text: list, queries_text: list, code_examples_vector: list, query_vectors: list, num_hash_tables: int|list, num_hash_functions: int, top_n: int, threshold: int = None, bucket_width: int = None) -> tuple[float, float]:
    """
    Executes Locality Sensitive Hashing (LSH) algorithms on queries and data vectors.

    Parameters:
        - output_path (str): output path to save results.
        - algorithm_type (str): type of LSH algorithm (accepted only 'la' or 'qa')
        - queries_type (str): type of queries (accepted only 'api' or 'nl')
        - code_examples_text (list): list of code examples dataset.
        - queries_text (list): list of queries dataset (API or NL).
        - code_examples_vector (list): list of code examples vectors embeddings.
        - query_vectors (list): list of query vectors embeddings.
        - num_hash_tables (int or list): number of hash tables.
        - num_hash_functions (int): number of hash functions.
        - top_n (int): number of recommended results (top n results).
        - threshold (int, optional): threshold for Query Aware LSH (default None).
        - bucket_width (int, optional): bucket width for Query Aware LSH (default None).

    Returns:
        - tuple containing two elements:
            - avg_creation_times (float): average creation times for each query.
            - avg_recommendation_times (float): average recommendation times for each query.
    """
    if algorithm_type.lower() not in ('rh', 'qa'):
        raise Exception('Invalid algorithm, choose between "rh" or "qa"!')
    if queries_type.lower() not in ('api', 'nl'):
        raise Exception('Invalid queries type, choose between "api" or "nl"!')

    
    original_stdout = sys.stdout

    # Create output folder.
    algorithm_output_path = os.path.join(output_path, f'{algorithm_type.lower()}_lsh/{queries_type.lower()}_queries/M={num_hash_tables}/k={num_hash_functions}/{f"w={bucket_width}/" if bucket_width else ""}')

    if not os.path.exists(algorithm_output_path):
        os.makedirs(algorithm_output_path)

    # Determine LSH algorithm type based.
    if algorithm_type.lower() == 'rh':
        alg = RandomHyperplaneLSH(num_hash_tables=num_hash_tables, num_hash_functions=num_hash_functions)
        print(f'Random Hyperplane LSH on {"API" if queries_type.lower() == "api" else "Natural Language"} queries (with M={num_hash_tables} and k={num_hash_functions}).')
    else:
        alg = QueryAwareLSH(num_hash_tables=num_hash_tables, num_hash_functions=num_hash_functions, threshold=threshold)
        print(f'Query Aware LSH on {"API" if queries_type.lower() == "api" else "Natural Language"} queries (with M={num_hash_tables}, k={num_hash_functions}, l={threshold} and w={bucket_width}).')

    # Initialize lists to store creation and recommendation times.
    creation_times: list = []
    recommendation_times: list = []

    for i, query_vector in enumerate(tqdm(query_vectors, desc="Processing queries", smoothing=0.05, dynamic_ncols=True)):
        # Save output results.
        output_result_path = os.path.join(algorithm_output_path, f'query_{str(i + 1)}/')

        if not os.path.exists(output_result_path):
            os.makedirs(output_result_path)
        
        sys.stdout = open(os.path.join(output_result_path, 'results.dat'), 'w', encoding='utf-8')
            
        nl = "\n"
        print(f'Hyperparameters:\n-M={num_hash_tables}\n-k={num_hash_functions}\n-N={top_n}{f"{nl}-l={threshold}{nl}-w={bucket_width}{nl}" if algorithm_type.lower() == "qa" else f"{nl}"}')

        # Execute LSH.
        if algorithm_type == "rh":
            recommended_samples_index, creation_time, recommendation_time = alg.lsh(code_examples_vector, query_vector, top_n=top_n)
        else:
            recommended_samples_index, creation_time, recommendation_time = alg.lsh(code_examples_vector, query_vector, top_n=top_n, bucket_width=bucket_width)
        print(f'\nRecommended samples code index: {recommended_samples_index}')

        # Save creation and recommendation times.
        creation_times.append(creation_time)
        recommendation_times.append(recommendation_time)

        # Save code examples results
        for j, recommended_sample_index in enumerate(recommended_samples_index):
            with open(os.path.join(output_result_path, f'{str(j + 1)} (sample_{str(recommended_sample_index)}).dat'), 'w', encoding='utf-8') as f:
                f.write(f"Query:\n\n{queries_text[i]}\n\n\n")
                f.write(f"Code Example {recommended_sample_index} (Top {str(j + 1)}):\n\n{code_examples_text[recommended_sample_index]}")

        sys.stdout.close()
        sys.stdout = None

    # Return to console.
    sys.stdout = original_stdout

    # Return average creation and recommendation times.
    return (np.average(creation_times), np.average(recommendation_times))

def execute_lsh(output_path: str, bert_model: str, queries_type: str, code_examples_text: list, queries_text: list, code_examples_vector: list, query_vectors: list, num_hash_tables: int|list, num_hash_functions: int, top_n: int, threshold: int = None, bucket_width: int = None) -> tuple[float, float]:
    """
    Executes Locality Sensitive Hashing (LSH) algorithms on queries and data vectors.

    Parameters:
        - output_path (str): output path to save results.
        - bert_model (str): type of BERT model used to generate embeddings (accepted only 'base' or 'fine-tuning').
        - queries_type (str): type of queries (accepted only 'api' or 'nl')
        - code_examples_text (list): list of code examples dataset.
        - queries_text (list): list of queries dataset (API or NL).
        - code_examples_vector (list): list of code examples vectors embeddings.
        - query_vectors (list): list of query vectors embeddings.
        - num_hash_tables (int or list): number of hash tables.
        - num_hash_functions (int): number of hash functions.
        - top_n (int): number of recommended results (top n results).
        - threshold (int, optional): threshold for Query Aware LSH (default None).
        - bucket_width (int, optional): bucket width for Query Aware LSH (default None).

    Returns:
        - tuple containing two elements:
            - avg_creation_times (float): average creation times for each query.
            - avg_recommendation_times (float): average recommendation times for each query.
    """
    if bert_model.lower() not in ('base', 'fine-tuning'):
        raise Exception('Invalid model, choose between "base" or "fine-tuning"!')
    if queries_type.lower() not in ('api', 'nl'):
        raise Exception('Invalid queries type, choose between "api" or "nl"!')

    
    original_stdout = sys.stdout

    # Create output folder.
    model_output_path = os.path.join(output_path, f'{bert_model.lower()}_model/{queries_type.lower()}_queries/M={num_hash_tables}/k={num_hash_functions}/{f"w={bucket_width}/" if bucket_width else ""}')

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # Determine LSH algorithm type based.
    if bert_model.lower() == 'base':
        alg = QueryAwareLSH(num_hash_tables=num_hash_tables, num_hash_functions=num_hash_functions, threshold=threshold)
        print(f'Query Aware LSH on {"API" if queries_type.lower() == "api" else "Natural Language"} queries (with M={num_hash_tables}, k={num_hash_functions}, l={threshold} and w={bucket_width}) using base BERT model.')
    else:
        alg = QueryAwareLSH(num_hash_tables=num_hash_tables, num_hash_functions=num_hash_functions, threshold=threshold)
        print(f'Query Aware LSH on {"API" if queries_type.lower() == "api" else "Natural Language"} queries (with M={num_hash_tables}, k={num_hash_functions}, l={threshold} and w={bucket_width}) using fine-tuned BERT model.')

    # Initialize lists to store creation and recommendation times.
    creation_times: list = []
    recommendation_times: list = []

    for i, query_vector in enumerate(tqdm(query_vectors, desc="Processing queries", smoothing=0.05, dynamic_ncols=True)):
        # Save output results.
        output_result_path = os.path.join(model_output_path, f'query_{str(i + 1)}/')

        if not os.path.exists(output_result_path):
            os.makedirs(output_result_path)
        
        sys.stdout = open(os.path.join(output_result_path, 'results.dat'), 'w', encoding='utf-8')
            
        print(f'Hyperparameters:\n-M={num_hash_tables}\n-k={num_hash_functions}\n-N={top_n}\n-l={threshold}\n-w={bucket_width}')

        # Execute LSH.
        recommended_samples_index, creation_time, recommendation_time = alg.lsh(code_examples_vector, query_vector, top_n=top_n, bucket_width=bucket_width)
        print(f'\nRecommended samples code index: {recommended_samples_index}')

        # Save creation and recommendation times.
        creation_times.append(creation_time)
        recommendation_times.append(recommendation_time)

        # Save code examples results
        for j, recommended_sample_index in enumerate(recommended_samples_index):
            with open(os.path.join(output_result_path, f'{str(j + 1)} (sample_{str(recommended_sample_index)}).dat'), 'w', encoding='utf-8') as f:
                f.write(f"Query:\n\n{queries_text[i]}\n\n\n")
                f.write(f"Code Example {recommended_sample_index} (Top {str(j + 1)}):\n\n{code_examples_text[recommended_sample_index]}")

        sys.stdout.close()
        sys.stdout = None

    # Return to console.
    sys.stdout = original_stdout

    # Return average creation and recommendation times.
    return (np.average(creation_times), np.average(recommendation_times))

def main():
    """
    Main function that executes the Locality Sensitive Hashing (LSH) algorithms on code examples and queries.

    This function loads text data and embeddings from files, defines hyperparameters, and executes LSH algorithms
    on both Natural Language queries and API queries.
    The function saves the output results to separate folders based on the hyperparameters and query types.

    The function does not return any value. It saves the output results to files instead.
    """
    output_path = './results (base vs fine-tuning)/'

    # Load text data.
    print('Loading text...\n')
    code_examples, nl_queries, api_queries = load_text('./datasets/code_examples.json', './datasets/queries.json')

    # Load embeddings.
    print('Loading embeddings...\n')
    np.random.seed(0)
    base_code_examples_vector, base_nl_query_vectors, base_api_query_vectors = load_embeddings('./embeddings (bert-base-uncased)/code_examples_embeddings.pkl', './embeddings (bert-base-uncased)/nl_queries_embeddings.pkl', './embeddings (bert-base-uncased)/api_queries_embeddings.pkl')
    ft_code_examples_vector, ft_nl_query_vectors, ft_api_query_vectors = load_embeddings('./embeddings (TSDAE + MNR)/code_examples_embeddings.pkl', './embeddings (TSDAE + MNR)/nl_queries_embeddings.pkl', './embeddings (TSDAE + MNR)/api_queries_embeddings.pkl')

    # Define hyperparameters.
    num_hash_tables = [2, 5, 10, 20, 30, 40, 50]
    num_hash_functions = 10
    top_n = 30
    threshold = 2
    bucket_width = 40

    # Transform num_hash_tables in a list if is an int value.
    if isinstance(num_hash_tables, int):
        num_hash_tables = [num_hash_tables]

    # Transform num_hash_functions in a list if is an int value.
    if isinstance(num_hash_functions, int):
        num_hash_functions = [num_hash_functions]

    # Transform bucket_width in a list if is an int value.
    if isinstance(bucket_width, int):
        bucket_width = [bucket_width]

    # Initialize lists to store weighted average of creation and recommendation times.
    base_avg_creation_times = []
    base_avg_recommendation_times = []
    ft_avg_creation_times = []
    ft_avg_recommendation_times = []

    for m in num_hash_tables:
        for k in num_hash_functions:
            for w in bucket_width:
                # Execute Qwery Aware LSH on API queries (base model).
                base_avg_creation_time_api, base_avg_recommendation_time_api = execute_lsh(output_path, 'base', 'api', code_examples, api_queries, base_code_examples_vector, base_api_query_vectors, m, k, top_n, threshold, w)

                # Execute Qwery Aware LSH on Natural Language queries (base model).
                base_avg_creation_time_nl, base_avg_recommendation_time_nl = execute_lsh(output_path, 'base', 'nl', code_examples, nl_queries, base_code_examples_vector, base_nl_query_vectors, m, k, top_n, threshold, w)
                
                # Calculate weighted average of creation and recommendation times.
                base_avg_creation_times.append(np.around(np.average([base_avg_creation_time_api, base_avg_creation_time_nl]), 3))
                base_avg_recommendation_times.append(np.around(np.average([base_avg_recommendation_time_api, base_avg_recommendation_time_nl]), 3))

                # Execute Qwery Aware LSH on API queries (fine-tuned model).
                ft_avg_creation_time_api, ft_avg_recommendation_time_api = execute_lsh(output_path, 'fine-tuning', 'api', code_examples, api_queries, ft_code_examples_vector, ft_api_query_vectors, m, k, top_n, threshold, w)

                # Execute Qwery Aware LSH on Natural Language queries (fine-tuned model).
                ft_avg_creation_time_nl, ft_avg_recommendation_time_nl = execute_lsh(output_path, 'fine-tuning', 'nl', code_examples, nl_queries, ft_code_examples_vector, ft_nl_query_vectors, m, k, top_n, threshold, w)

                # Calculate weighted average of creation and recommendation times.
                ft_avg_creation_times.append(np.around(np.average([ft_avg_creation_time_api, ft_avg_creation_time_nl]), 3))
                ft_avg_recommendation_times.append(np.around(np.average([ft_avg_recommendation_time_api, ft_avg_recommendation_time_nl]), 3))

    # Plot creation time.
    plot_model_compare(num_hash_tables=num_hash_tables, base_times=base_avg_creation_times, ft_times=ft_avg_creation_times, title='LSH Creation Time', labels=['Number of Hash Tables', 'Creation Time (s)'], output_path=os.path.join(output_path, 'creation_time.png'))

    # Plot recommendation time.
    plot_model_compare(num_hash_tables=num_hash_tables, base_times=base_avg_recommendation_times, ft_times=ft_avg_recommendation_times, title='LSH Recommendation Time', labels=['Number of Hash Tables', 'Recommendation Time (s)'], output_path=os.path.join(output_path, 'recommendation_time.png'))


if __name__ == '__main__':
    main()