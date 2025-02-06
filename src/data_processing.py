from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def load_and_preprocess_data(
    filepath: str, start_token: str = "!", end_token: str = "."
) -> List[Tuple[str, str]]:
    """
    Load text from a file and preprocess them into bigrams with specified start and end tokens.

    This function reads a file where each line contains a word followed by additional data
    (typically two numbers), all separated by spaces. It processes each word by adding specified
    start and end tokens, then creates bigrams for each processed word. Remember transform the
    letters to lowercase.

    Note: It is expected that each line in the file contains a word followed by two numerical
    data elements, all separated by spaces. The last two elements shall be ignored.

    Args:
        filepath: str. Path to the text file.
        start_token: str. A character used as the start token for each word.
        end_token: str. A character used as the end token for each word.

    Returns:
        List[Tuple[str, str]]. A list of bigrams, where each bigram is a tuple of two characters.
    """
    with open(filepath, "r") as file:
        lines: List[str] = file.read().splitlines()  # list of rows of the file

    bigrams: List[Tuple[str, str]] = []
    for line in lines:
        list_line: List[str] = line.lower().split()
        # remove lasts 2 elements
        list_line.pop()
        list_line.pop()
        list_new_line: List[str] = [start_token]
        for word in list_line:
            list_new_line += list(word)
        list_new_line.append(end_token)
        for i in range(len(list_new_line)-1):
            bigram = (list_new_line[i], list_new_line[i+1])
            bigrams.append(bigram)
    print(bigrams[2])
    return bigrams


def char_to_index(alphabet: str, start_token: str, end_token: str) -> Dict[str, int]:
    """
    Create a dictionary mapping each character in the alphabet to an index.

    Args:
        alphabet: str. The alphabet string containing unique characters.
        start_token: str. The start token to be added at the beginning of the index.
        end_token: str. The end token to be added at the end of the index.

    Returns:
        Dict[str, int]: A dictionary mapping each character, including start and end tokens, to an index.
    """
    # Create a dictionary with start token at the beginning and end token at the end
    char_to_idx: Dict[str, int] = {}
    char_to_idx[start_token] = 0
    idx = 1
    for char in alphabet:
        char_to_idx[char] = idx
        idx += 1
    if idx in char_to_idx.keys():  # To avoid deleting the last character of the alphabet
        idx += 1
    char_to_idx[end_token] = idx
    return char_to_idx


def index_to_char(char_to_index: Dict[str, int]) -> Dict[int, str]:
    """
    Create a dictionary mapping each index to the corresponding character in the alphabet.

    Args:
        char_to_index: Dict[str, int]. A dictionary mapping characters to indices.

    Returns:
        Dict[int, str]: A dictionary mapping each index back to its corresponding character.
    """
    # Reverse the char_to_index mapping
    idx_to_char: Dict[int, str] = {}
    for char, idx in char_to_index.items():
        idx_to_char[idx] = char
    return idx_to_char


def count_bigrams(
    bigrams: List[Tuple[str, str]], char_to_idx: Dict[str, int]
) -> torch.Tensor:
    """
    Count the frequency of each bigram in the given list using PyTorch.

    This function is designed to work with the output of the 'load_and_preprocess_data'
    function, which processes a text file into a list of bigrams. It counts the frequency
    of each bigram within the provided list using a PyTorch tensor.

    Args:
        bigrams: List[Tuple[str, str]]. A list of bigrams, typically the output from
                 'load_and_preprocess_data'.
        char_to_idx: Dict[str, int]. A dictionary mapping characters to their indices in the alphabet.

    Returns:
        torch.Tensor. A 2D tensor where each cell (i, j) represents the count of the bigram
        formed by the i-th and j-th characters in the alphabet.
    """
    # Initialize a 2D tensor for counting bigrams
    bigram_counts: torch.Tensor = torch.zeros((len(char_to_idx), len(char_to_idx)), dtype=torch.int32)
    # Iterate over each bigram and update the count in the tensor
    for char1, char2 in bigrams:
        try:
            bigram_counts[char_to_idx[char1], char_to_idx[char2]] += 1
        except KeyError:
            # We are omitting those characters that are not in the alphabet
            print(f"{char1} or {char2} not in the alphabet")
    return bigram_counts


def plot_bigram_counts(bigram_counts: torch.Tensor, idx_to_char: Dict):
    """
    Plot the bigram counts in a heatmap style with annotations.

    Args:
        bigram_counts: torch.Tensor. A 2D tensor of bigram counts.
        idx_to_char: Dict. A dictionary mapping indices to characters.
    """
    plt.figure(figsize=(16, 16))
    plt.imshow(bigram_counts, cmap="Blues")

    for i in range(bigram_counts.shape[0]):
        for j in range(bigram_counts.shape[1]):
            char_str: str = idx_to_char[i] + idx_to_char[j]
            plt.text(j, i, char_str, ha="center", va="bottom", color="gray")
            plt.text(
                j, i, bigram_counts[i, j].item(), ha="center", va="top", color="gray"
            )

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Define the path to your file
    file_path: str = "data/nombres_raw.txt"

    # Define the alphabet (ensure it covers all characters in your data)
    alphabet: str = "abcdefghijklmnopqrstuvwxyz "

    start_token: str = "-"
    end_token: str = "."

    # Create a mapping from characters to indices
    char_to_idx: Dict[str, int] = char_to_index(alphabet, start_token, end_token)

    # Load and preprocess the data to get bigrams
    bigrams: List[Tuple[str, str]] = load_and_preprocess_data(
        file_path, start_token=start_token, end_token=end_token
    )

    # Count the bigrams
    bigram_counts: torch.Tensor = count_bigrams(bigrams, char_to_idx)

    # Create a mapping from indices to characters (reverse of char_to_index)
    idx_to_char: Dict[int, str] = index_to_char(char_to_idx)

    # Plot the bigram counts
    plot_bigram_counts(bigram_counts, idx_to_char)

