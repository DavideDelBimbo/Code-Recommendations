import os
import json
import re
import pickle
from tqdm import tqdm

import torch

from transformers import BertTokenizer, BertModel
import torch

class EmbeddingsGenerator:
    def __init__(self, model_name_or_path: str = 'bert-base-uncased'):
        """
        Initializes the EmbeddingsGenerator.

        Parameters:
            - model_name_or_path (str, optional): name or path of the RobertaModel model to use (default 'bert-base-uncased').
        """
        self.model_name_or_path: str = model_name_or_path
        
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model: BertModel = BertModel.from_pretrained(self.model_name_or_path)
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)

        self.code_examples: list = None
        self.nl_queries: list = None
        self.api_queries: list = None

        self.code_examples_embeddings: list = []
        self.nl_queries_embeddings: list = []
        self.api_queries_embeddings: list = []

    # Function to load code examples.
    def load_code_examples(self, file_path='./datasets/code_examples.json'):
        """
        Loads code examples from a JSON file.

        Parameters:
            - file_path (str, optional): path to the JSON file containing code examples (default './datasets/code_examples.json').
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.code_examples = json.load(f)

    # Function to load queries.
    def load_queries(self, file_path: str = './datasets/queries.json'):
        """
        Loads and preprocesses queries from a text file, separating Natural Language and API queries.

        Parameters:
            - file_path (str, optional): path to the text file containing queries (deafult './datasets/queries.json').
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data: dict = json.load(f)

            self.nl_queries = data["natural_language"]
            self.api_queries = data["api"]

    # Function to generate embeddings.
    def generate_embeddings(self):
        """
        Generates embeddings for code examples, Natural Language queries and API queries using the pre-trained model.
        """
        # Function to remove preprocess code example (removes space and tags).
        def __preprocess_code_example(code_example: str) -> str:
            # Remove tag from code examples.
            code_example = re.sub(r"[\t\n]", "", code_example)
            code_example = re.sub(r"\s+", " ", code_example)

            return code_example

        # Get embeddings vector for code examples.
        for code_example in tqdm(self.code_examples, desc="Embeddings code examples", smoothing=0.05, dynamic_ncols=True):
            # Preprocess code example.
            preprocessed_code_example: str = __preprocess_code_example(code_example)

            # Get embeddings vector for code examples.
            code_example_tokens: torch.Tensor = self.tokenizer.encode(preprocessed_code_example, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt")

            with torch.no_grad():
                # Predict the embeddings vector for the code example.
                output = self.model(code_example_tokens.to(self.device))

                # Calculate the mean of the embeddings vector for each token.
                context_embeddings = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

                # Append the embeddings vector to the list of code examples embeddings.
                self.code_examples_embeddings.append(context_embeddings)

        # Get embeddings vector for Natural Language queries.
        for nl_query in tqdm(self.nl_queries, desc="Embeddings NL queries", smoothing=0.05, dynamic_ncols=True):
            # Get embeddings vector for Natural Language query.
            nl_query_tokens: torch.Tensor = self.tokenizer.encode(nl_query, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt")

            with torch.no_grad():
                # Predict the embeddings vector for the Natural Language query.
                output = self.model(nl_query_tokens.to(self.device))

                # Calculate the mean of the embeddings vector for each token.
                context_embeddings = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

                # Append the embeddings vector to the list of Natural Language queries embeddings.
                self.nl_queries_embeddings.append(context_embeddings)

        # Get embeddings vector for API queries.
        for api_query in tqdm(self.api_queries, desc="Embeddings API queries", smoothing=0.05, dynamic_ncols=True):
            # Get embeddings vector for API query.
            api_query_tokens: torch.Tensor = self.tokenizer.encode(api_query, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt")

            with torch.no_grad():
                # Predict the embeddings vector for the API query.
                output = self.model(api_query_tokens.to(self.device))

                # Calculate the mean of the embeddings vector for each token.
                context_embeddings = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                # Append the embeddings vector to the list of API queries embeddings.
                self.api_queries_embeddings.append(context_embeddings)
                
    # Function to save embeddings generated vectors.
    def save_embeddings(self, output_path: str = './embeddings/'):
        """
        Saves the generated embeddings to pickle files.

        Parameters:
            - output_path (str, optional): path to save embeddings pickle files (default './embeddings/')
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'code_examples_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.code_examples_embeddings, f)

        with open(os.path.join(output_path, 'nl_queries_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.nl_queries_embeddings, f)

        with open(os.path.join(output_path, 'api_queries_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.api_queries_embeddings, f)

def main():
    embeddings_generator = EmbeddingsGenerator(model_name_or_path='./model (TSDAE + MNR)')
    embeddings_generator.load_code_examples()
    embeddings_generator.load_queries()
    embeddings_generator.generate_embeddings()
    embeddings_generator.save_embeddings(output_path='./embeddings (TSDAE + MNR)')


if __name__ == '__main__':
    main()

