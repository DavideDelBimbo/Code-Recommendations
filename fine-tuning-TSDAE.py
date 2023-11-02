import os
import csv
import json
import time
import re
import random
import math
import socket
from datetime import datetime
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn import model_selection

from sentence_transformers import SentenceTransformer
from sentence_transformers import models, datasets, losses, evaluation, util

class CodeExampleTrainer:
    def __init__(self, model_name_or_path: str = 'bert-base-uncased', seed: int = 0, log_dir: str = './logs/') -> None:
        """
        Initialize the CodeExampleTrainer.

        Parameters:
            - model_name_or_path (str): name or path of the SentenceTransformer model to use (default 'bert-base-uncased').
            - seed (int): random seed for reproducibility (default 0).
            - log_dir (str, optional): path for TensorBoard logs (default './logs/').
        """
        self.model_name_or_path: str = model_name_or_path
        self.seed: int = seed
        
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.code_examples: list = []

        self.train_set = None
        self.train_dataloader: DataLoader = None

        self.dev_set = None
        self.dev_dataloader: DataLoader = None

        self.model: SentenceTransformer = None
        self.train_loss: LossTrain  = None
        self.evaluator: LossEvaluator = None

        self.time_start: float = time.time()
        self.best_score: float = float('inf')

        self.log_dir = os.path.join(log_dir, f"{datetime.now().strftime('%d-%b %H-%M-%S')} {socket.gethostname()}") if log_dir else None
    
    # Function to set random seed for reproducibility.
    def __set_seed(self) -> None:
        """
        Set seed for random torch operations.
        """
        # Set the seed for general torch operations.
        torch.manual_seed(self.seed)
        # Set the seed for CUDA torch operations (GPU operations).
        torch.cuda.manual_seed(self.seed)
        # Set the seed for random operations.
        random.seed(self.seed)
        # Set the seed for numpy operations.
        np.random.seed(self.seed)

    # Function to estimate size of a batch.
    def __batch_size(self, batch) -> float:
        """
        Function to calculate size of a batch.

        Parameters:
            - batch: batch of sample from DataLoader.

        Returns:
            - float: size of a batch (Gb).
        """        
        noised_features = [b for b in batch[0][0].values()]
        original_features = [b for b in batch[0][1].values()]
        labels = batch[1]
            
        noised_features_size: int = sum(tensor.numel() for tensor in noised_features)
        original_features_size: int = sum(tensor.numel() for tensor in original_features)
        labels_size: int = labels.numel()

        total_size: int = noised_features_size + original_features_size + labels_size
        
        return round(total_size / (1024 * 3), 2)
    
    # Function to load model.
    def load_model(self) -> None:
        """
        Load SentenceTransformer model.
        """
        from transformers import logging
        logging.set_verbosity_error()

        # Use huggingface/transformers model (like BERT, RoBERTa) for mapping tokens to embeddings.
        # Model is cached in "C:\Users\username\.cache\huggingface\hub".
        word_embedding_model = models.Transformer(self.model_name_or_path, max_seq_length=256)

        # Apply cls pooling to get one fixed sized sentence vector.
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')

        # Define sentence transformer model.
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # Load model on GPU if available.
        self.model.to(self.device)

    # Function to load dataset.
    def load_dataset(self, file_path='./datasets/code_examples.json') -> None:
        """
        Load dataset of questions and code examples from a JSON file.
        
        Parameters:
            - file_path (str, optional): path to code examples dataset JSON file (default './datasets/code_examples.json').
        """
        start = time.time()
        print("Loading dataset...")

        # Load JSON dataset.
        with open(file_path, 'r', encoding='utf-8') as f:
            self.code_examples = json.load(f)

        print(f"Dataset loaded in {time.time() - start}")

    # Function to preprocessing dataset.
    def preprocess_data(self) -> None:
        """
        Cleans dataset and create training pairs using query and code examples.
        """
        # Preprocess code examples.
        for i, code_example in tqdm(enumerate(self.code_examples), total=len(self.code_examples), desc='Preprocessing', smoothing=0.05, dynamic_ncols=True):
            # Remove tag from code examples.
            code_example = re.sub(r"[\t\n]", "", code_example)
            code_example = re.sub(r"\s+", " ", code_example)

            # Replace cleaned code example to dataset.
            self.code_examples[i] = code_example

    # Function to evaluate model during training.
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback, min_delta = 0.001) -> None:
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if score < (self.best_score - min_delta):
                self.best_score = score
                if save_best_model:
                    self.model.save(output_path)
                    with open(os.path.join(output_path, "best_score.txt"), "w") as text_file:
                        text_file.write(f"Trining on {len(self.train_dataloader)} batches and {len(self.train_set)} samples\n")
                        text_file.write(f"Evaluated on {len(self.dev_dataloader)} batches and {len(self.dev_set)} samples\n")
                        text_file.write(f"Saved at time: {datetime.now().strftime('%d-%b %H-%M-%S')}\n")
                        text_file.write(f"Best score: {score}\n")
                        text_file.write(f"\t-Epoch: {epoch}\n")
                        text_file.write(f"\t-Steps: {steps if steps != -1 else len(self.train_dataloader)}\n")
                        text_file.write(f"\t-Training time: {(time.time() - self.time_start) / 60} minutes\n")
            if callback is not None:
                callback(score, epoch, steps)
    
    # Function to fine-tune model.
    def fine_tune_model(self, batch_size: int = 4, epochs: int = 5, evaluation_steps: int = 1000, output_path: str = './model/') -> None:
        """
        Performs fine-tuning of the model.

        Parameters:
            - batch_size (int, optional): size of batch of data (default 4).
            - epochs (int, optional): number of epochs for fine-tuning (deafult 5).
            - evaluation_steps (int, optional): number of steps between each evaluation (default 1000).
            - output_path (int, optional): path to save the fine-tuned model (default './model/').
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # Set seed to reproducibility.
        self.__set_seed()


        # Split dataset into training set and dev set.
        train_split, dev_split = model_selection.train_test_split(self.code_examples, test_size=0.2, random_state=self.seed, shuffle=True)
        

        # Create denoising dataset that adds noise on training dataset split (remove words from example code).
        self.train_set = datasets.DenoisingAutoEncoderDataset(train_split)
        # Create DataLoader from training dataset.
        self.train_dataloader: DataLoader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.model.smart_batching_collate)


        # Create denoising dataset that adds noise on development dataset split (remove words from example code).
        self.dev_set = datasets.DenoisingAutoEncoderDataset(dev_split)
        # Create DataLoader from development dataset.
        self.dev_dataloader: DataLoader = DataLoader(self.dev_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.model.smart_batching_collate)

        
        # Define loss.
        self.train_loss = LossTrain(model=self.model, decoder_name_or_path=self.model_name_or_path, tie_encoder_decoder=True, evaluation_steps=evaluation_steps, epoch_steps=len(self.train_dataloader), log_dir=self.log_dir)
        # Load loss function on GPU if available.
        self.train_loss.to(self.device)


        # Define evaluator.
        self.evaluator = LossEvaluator(dataloader=self.dev_dataloader, loss=self.train_loss, epoch_steps=len(self.train_dataloader), show_progress_bar=True, output_name='dev', log_dir=self.log_dir)
        # Set model training mode.
        self.model._eval_during_training = self._eval_during_training


        # Define EarlyStopping callback.
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, output_path=os.path.join(output_path, 'best_score.txt') if output_path else None)


        # Define warmup for the first 10% of training steps.
        warmup_steps = math.ceil(len(self.train_dataloader) * epochs * 0.1)


        # Execute fine-tuning.
        self.model.fit(
            train_objectives = [(self.train_dataloader, self.train_loss)],
            epochs = epochs,
            warmup_steps = warmup_steps,
            evaluator = self.evaluator,
            evaluation_steps = evaluation_steps,
            callback = early_stopping,
            show_progress_bar = True,
            output_path = output_path,
        )


class LossTrain(losses.DenoisingAutoEncoderLoss):
    def __init__(self, evaluation_steps: int, epoch_steps: int, log_dir: str = None, *args, **kwargs) -> None:
        """
        Initialize the loss function for training.

        Parameters:
            - evaluation_steps (int): number of steps between each evaluation.
            - epoch_steps (int): number of steps between each epoch.
            - log_dir (str, optional): path for TensorBoard logs (default None).
        """
        super().__init__(*args, **kwargs)

        self.evaluation_steps: int = evaluation_steps
        self.epoch_steps: int = epoch_steps

        self.global_steps: int = 0
        self.current_step: int = 0
        self.current_epoch: int = 0

        self.is_training: bool = True

        if log_dir:
            self.log_dir = os.path.join(log_dir, "loss", "training")
            self.logs_writer: SummaryWriter = SummaryWriter(log_dir=self.log_dir)

    def forward(self, sentence_features, labels) -> float:
        loss_value = super().forward(sentence_features, labels)

        # Log the loss value on TensorBoard if in training mode.
        if self.is_training and self.log_dir:
            self.global_steps += 1
            self.current_step += 1
            # Reset current step and increment current epoch if reached the end of an epoch.
            if self.current_step % self.epoch_steps == 0:
                self.current_step = 0
                self.current_epoch += 1
            # Log the loss value on TensorBoard if reached the number of steps between each evaluation or the end of an epoch.
            if self.current_step % self.evaluation_steps == 0 or self.current_step % self.epoch_steps == 0:
                self.logs_writer.add_scalar('Loss', loss_value, self.global_steps)
        
        return loss_value


class LossEvaluator(evaluation.SentenceEvaluator):
    def __init__(self, dataloader: DataLoader, loss: LossTrain, epoch_steps: int, show_progress_bar: bool = False, write_csv: bool = True, output_name: str = '', log_dir: str = None) -> None:
        """
        Evaluate a model based on the loss function. The results are written in a CSV and Tensorboard logs.

        Parameters:
            - dataloader (DataLoader): DataLoader object.
            - loss (LossTrain, optional): loss module object (deafult None).
            - epoch_steps (int): number of steps between each epoch.
            - show_progress_bar (bool, optional): if true, prints a progress bar (deafult False).
            - write_csv (bool, optional): write results to a CSV file (deafult True).
            - output_name (str, optional): name for the output file (default '').
            - log_dir (str, optional): path for TensorBoard logs (default None).
        """
        self.dataloader: DataLoader = dataloader
        self.loss: LossTrain = loss

        self.epoch_steps: int = epoch_steps

        self.show_progress_bar: bool = show_progress_bar
        self.write_csv: bool = write_csv
        self.output_name: str = output_name

        # Load model on GPU if available.
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        loss.to(self.device)

        self.csv_file: str = "loss_evaluation" + ("_" + output_name if output_name else '') + "_results.csv"
        self.csv_headers: list = ["epoch", "steps", "loss"]

        if log_dir:
            self.log_dir = os.path.join(log_dir, "loss", "validation")
            self.logs_writer: SummaryWriter = SummaryWriter(log_dir=self.log_dir)
    
    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        # Set evaulation mode.
        self.loss.eval()
        self.loss.is_training = False

        # Initialize loss value.
        loss_value = 0

        # Create iterator for the DataLoader.
        dataloader_iterator = iter(self.dataloader)

        with torch.no_grad():
            for _ in trange(len(self.dataloader), desc="Evaluation", smoothing=0.05, disable=not self.show_progress_bar, dynamic_ncols=True, leave=False):
                data_features, labels = next(dataloader_iterator)

                # Load data on GPU if available.
                data_features: list = list(map(lambda batch: util.batch_to_device(batch, self.device), data_features))
                labels: torch.Tensor = labels.to(self.device)

                # Update loss value.
                loss_value += self.loss(data_features, labels).item()

        # Calculate the average loss over the number of batches in the dataloader.
        final_loss = loss_value / len(self.dataloader) # len(self.dataloader) = num_batches

        # Save results on output file.
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)

            with open(csv_path, newline='', mode='a' if output_file_exists else 'w', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Writes the header if the file doesn't yet exist.
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                # Writes the evaluation related to the training iteration.
                writer.writerow([epoch, steps if steps != -1 else self.epoch_steps, final_loss])

        if self.log_dir:
            # Log the evaluated loss.
            self.logs_writer.add_scalar('Loss', final_loss, self.loss.global_steps)

        # Reset gradients and return to training mode.
        self.loss.zero_grad()
        self.loss.train()
        self.loss.is_training = True

        return final_loss


class EarlyStopping:
    def __init__(self, patience, min_delta: float = 0, output_path: str = None) -> None:
        """
        Early stopping to stop the training when the loss does not improve after a given number of evaluations.

        Parameters:
            - patience (int): Number of evaluations with no improvement after which training will stop.
            - min_delta (float, optional): Minimum change in the score to be considered as an improvement (default 0).
            - output_path (str, optional): output path to save early stopping logs (default None).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.output_path = output_path

        self.best_score = None
        self.wait = 0

    def __call__(self, score, epoch, steps) -> None:
        """
        Callback function that checks for early stopping.

        Parameters:
            - score (float): The evaluation score, which could be a loss or any other metric.
            - epoch (int): The current epoch number.
            - steps (int): The current step or iteration number within the epoch.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < (self.best_score - self.min_delta):
            self.wait = 0
            self.best_score = score
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            # Stop training.
            if self.output_path:
                with open(self.output_path, "a") as text_file:
                    text_file.write(f"Early stopping at Epoch {epoch} and Step {steps} with Score {score}.\n")
            raise Exception(f"Early stopping at Epoch {epoch} and Step {steps} with Score {score}.")

def main():
    code_trainer = CodeExampleTrainer(model_name_or_path='bert-base-uncased', log_dir='./logs/')
    code_trainer.load_model()
    code_trainer.load_dataset()
    code_trainer.preprocess_data()
    code_trainer.fine_tune_model(batch_size=12, epochs=100, evaluation_steps=1000, output_path='./model/')


if __name__ == '__main__':
    main()