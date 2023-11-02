import os
import json
import re
import html
from tqdm import tqdm

import xml.etree.cElementTree as cET

class DataProcessor:
    def __init__(self, xml_file_path: str, threshold: int = 2) -> None:
        """
        Initializes the XMLDataProcessor object with the provided XML file path.

        Parameters:
            - xml_file_path (str): path to the XML dataset file to be processed.
            - threshold (int, optional): threshold score for code examples to be included (default 2).
        """
        # Path to the XML dataset file.
        self.xml_file_path: str = xml_file_path
        # Threshold value to include code example.
        self.threshold: int = threshold

        # Initialize dictionaries to store posts and code examples.
        self.posts: dict = {}
        self.code_examples: list = []


    # Function for parsing the XML dataset.
    def __parse_large_xml(self) -> None:
        """
        Parses the large XML dataset using an iterator.

        Yields:
            - elem: The current XML element being parsed.
        """
        # Open file with UTF-8 encoding.
        xml_file = open(self.xml_file_path, 'r', encoding='utf-8')

        # Create an XML tree iterator.
        context = cET.iterparse(xml_file, events=('end',))

        # Iterate through the XML elements.
        for _, elem in context:
            # Generate the current element.
            yield elem

            # Clear the element from memory to free up resources.
            elem.clear()

        xml_file.close()

    # Function to filter Java code examples.
    def __filter_code_examples(self, text: str, score: int) -> list:
        """
        Filters Java code examples from the given text based on specified criteria.

        Parameters:
            - text (str): input text containing code examples.
            - score (int): score associated with the code example.

        Returns:
            - filtered_code_examples (list of str): list of filtered code examples from text.
        """
        # Define the regex pattern to find code examples enclosed in <pre><code> and </code></pre> tags.
        pattern = r'<pre([^>]*)><code>(.*?)<\/code><\/pre>'
        
        # Find all matches of the pattern in the given text (si ricava solo il testo compreso tra i tag <code> e </code>).
        matches = re.finditer(pattern, text, re.DOTALL)
        code_examples = [match.group(2) for match in matches]

        with open('./dictionaries/java_keywords.txt', 'r', encoding='utf-8') as f:
            java_keywords: list = f.read().split("\n")

        with open('./dictionaries/sql_keywords.txt', 'r', encoding='utf-8') as f:
            sql_keywords: list = f.read().split("\n")

        # Filter the code examples based on specified criteria.
        filtered_code_examples = [
            code for code in code_examples 
            if len(re.sub(r'[\s\t\n]', '', code)) >= 100  # Exclude examples with less than 100 characters.
            and any(re.search(rf'\b{re.escape(java_keyword)}\b', code) for java_keyword in java_keywords) # Example must contains at least one java keyword.
            and not code.strip().startswith(('<', '>', '#', '@', '.', '-', '_', '$', '/', '*', '+', '='))  # Exclude examples starting with special characters (i.e. <, /, @).
            and not any(re.search(rf'\b{re.escape(sql_keyword)}\b', code) for sql_keyword in sql_keywords) # Exclude example containing SQL keywords.
            and not re.search(r'Exception(.*?)(\s+at.+)', code) # Exclude StackTraces' posts.
            and score >= self.threshold  # Exclude posts with a score lower than the threshold.
        ]

        return filtered_code_examples


    # Function to extract code examples from XML dataset.
    def process_xml_data(self, stop_iteration: int = None) -> None:
        """
        Processes the XML data, extracting relevant information and code examples.

        This function iterates through the XML elements and extracts code examples from Java related posts.

        Parameters:
            - stop_iteration (int, optional): stop parsing at assigned line.
        """
        with open('./dictionaries/excluding_tags.txt', 'r', encoding='utf-8') as f:
            excluding_tags_list: list = f.read().split("\n")
        
        i = 0
        # Iterate through the parsed XML elements.
        for post in tqdm(self.__parse_large_xml(), total=57_721_549, desc="Parsing XML", smoothing=0.05, dynamic_ncols=True):
            if i == stop_iteration:
                break
            try:
                # Get post's id.
                id = post.attrib['Id']

                # Check if the post is a valid Java question.
                if (post.attrib['PostTypeId'] == '1' # Type 1 indicates a question.
                    and '<java>' in post.attrib['Tags'] # Post must contains '<java>' tag.
                    and not any(f'<{tag}>' in post.attrib['Tags'] for tag in excluding_tags_list) # Post doesn't contain other programming languages tag (i.e. '<c>' or '<javascript>' tags).
                    and 'AcceptedAnswerId' in post.attrib # Post contains an accepted answer.
                    ):

                    # Store some post attributes.
                    self.posts[id] = {
                        'Query': html.unescape(post.attrib['Body']),
                        'AcceptedAnswerId': post.attrib['AcceptedAnswerId']
                    }
                
                # Check if the post is an accepted answer.
                elif (post.attrib['PostTypeId'] == '2'): # Type 2 indicates an answer.
                    # Get id of the parent question post.
                    parent_id = post.attrib['ParentId']

                    # Check if the answer has a parent question and if it is the accepted answer.
                    if (parent_id in self.posts and self.posts[parent_id]['AcceptedAnswerId'] == id):
                        # Filter out and get code examples from the accepted answer.
                        code_examples = self.__filter_code_examples(html.unescape(post.attrib['Body']), int(post.attrib['Score']))

                        # Save the code examples or remove the post if it didn't find them.
                        if code_examples:
                            self.posts[parent_id]['CodeExamples'] = code_examples
                            self.code_examples += code_examples
                        else:
                            self.posts.pop(parent_id)

            except Exception as e:
                print("An error occurred:", e)

                # Handle exceptions by continuing to the next iteration.
                pass
            
            i += 1

    # Function to save processed dataset (resulting Java code examples that satisfied some criteria).
    def save_data_to_json(self, output_path: str = './datasets', dataset_file: str = 'dataset', code_examples_file: str = 'code_examples', save_dataset: bool = True) -> None:
        """
        Saves the processed data to JSON files.

        Parameters:
            - ouput_path (str, optional): path to save ouput files (deafult './datasets')
            - dataset_file (str, optional): name of the JSON file to save dataset (default 'dataset').
            - code_examples_file (str, optional): name of the JSON file to save code examples list (default 'code_examples').
            - save_dataset (bool, optional): flag to save entire JSON dataset results.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if save_dataset:
            with open(os.path.join(output_path, dataset_file + '.json'), 'w', encoding='utf-8') as f:
                json.dump(self.posts, f, indent=2)
                print('Dataset saved successfully to file:', dataset_file)
        
        with open(os.path.join(output_path, code_examples_file  + '.json'), 'w', encoding='utf-8') as f:
            json.dump(self.code_examples, f, indent=2)
            print('Code examples saved successfully to file:', code_examples_file)
    
def main():
    xml_file_path = ".\stackexchange_20221206\Posts.xml"
    processor = DataProcessor(xml_file_path=xml_file_path)

    processor.process_xml_data()
    processor.save_data_to_json()


if __name__ == '__main__':
    main()