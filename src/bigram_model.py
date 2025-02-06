import torch
from typing import Dict, List, Tuple


def bigrams_count_to_probabilities(
    bigram_counts: torch.Tensor, smooth_factor: int = 0
) -> torch.Tensor:
    """
    Convert bigram counts to a probability distribution.

    This function normalizes the counts of bigrams to create a probability distribution
    for each starting character, representing the likelihood of each subsequent character.

    Args:
        bigram_counts: torch.Tensor. A 2D tensor where each cell (i, j) contains the count
                       of the bigram formed by the i-th and j-th characters in the alphabet.
        smooth_factor: int. A value to add to each bigram count for smoothing purposes.

    Returns:
        torch.Tensor. A 2D tensor where each row is a normalized probability distribution,
        indicating the likelihood of each character following the character corresponding
        to the row index.
    """
    bigram_probabilities: torch.Tensor = torch.zeros(bigram_counts.shape, dtype=torch.float32)

    # Normalize each row to sum to 1, converting counts to probabilities, remember to add smooth_factor
    smooth_counts = bigram_counts + smooth_factor
    total_sum = smooth_counts.sum(dim=1, keepdim=True)
    total_sum[total_sum == 0] = 1.0
    bigram_probabilities = smooth_counts / total_sum

    # No probability should be less or equal than 0
    for idx, row in enumerate(bigram_probabilities):
        if row.sum() <= 0:
            bigram_probabilities[idx] = torch.ones_like(row) / row.size(0)
    return bigram_probabilities

def calculate_neg_mean_log_likelihood(
    words: List[str],
    bigram_probabilities: torch.tensor,
    char_to_index: Dict[str, int],
    start_token: str = "<S>",
    end_token: str = "<E>",
) -> float:
    """
    Calculate the negative mean log likelihood of a list of words based on bigram probabilities.

    This function computes the negative mean log likelihood for a list of words using the provided bigram
    probability matrix. Each word's log likelihood is calculated using the 'calculate_log_likelihood'
    function, and the negative mean of these log likelihoods is returned.

    Args:
        words: List[str]. A list of words for which to calculate the mean log likelihood.
        bigram_probabilities: torch.Tensor. A 2D tensor representing the probability of each bigram.
        char_to_index: Dict. A dictionary mapping characters to their indices in the probability matrix.
        start_token: str. The character that denotes the start of a word. Shall be a single character.
        end_token: str. The character that denotes the end of a word. Shall be a single character.

    Returns:
        float. The negative mean log likelihood of the list of words.
    """
    # Initialize total log likelihood
    total_log_likelihood: torch.tensor = torch.zeros((len(words)), dtype=torch.float32)

    # Calculate the log likelihood for each word and accumulate
    for i, word in enumerate(words):
        total_log_likelihood[i] = calculate_log_likelihood(word, bigram_probabilities, char_to_index, start_token, end_token)


    # Calculate and return the negative mean log likelihood
    mean_log_likelihood: float = torch.mean(total_log_likelihood)
    return -mean_log_likelihood


def sample_next_character(
    current_char_index: int,
    probability_distribution: torch.Tensor,
    idx_to_char: Dict[int, str],
) -> str:
    """
    Sample the next character based on the current character index and probability distribution.

    Args:
        current_char_index: int. Index of the current character.
        probability_distribution: torch.Tensor. A 2D tensor of character probabilities.
        idx_to_char: Dict. Mapping from character indices to characters.

    Returns:
        str. The next character sampled based on the probability distribution.
    """
    # Get the probability distribution for the current character
    current_probs: torch.Tensor[float] = probability_distribution[current_char_index,:]
    if current_probs.sum() == 0:
        current_probs = torch.ones_like(current_probs) / current_probs.size(0)
    # Sample an index from the distribution using the torch.multinomial function
    next_char_index: int = torch.multinomial(current_probs, num_samples=1).item()
    # Map the index back to a character
    next_char: str = idx_to_char[next_char_index]
    return next_char


def generate_name(
    start_token: str,
    end_token: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    bigram_probabilities: torch.Tensor,
    max_length: int = 15,
) -> str:
    """
    Generate a new name based on the bigram probabilities.

    This function starts from the start token and iteratively samples the next character
    based on the current character's probability distribution until it reaches the end token
    or the maximum length.

    Args:
        start_token: str. The start token indicating the beginning of a name.
        end_token: str. The end token indicating the end of a name.
        char_to_idx: Dict[str, int]. A mapping from characters to their indices.
        idx_to_char: Dict[int, str]. A mapping from indices back to characters.
        bigram_probabilities: torch.Tensor. A 2D tensor representing the bigram probabilities.
        max_length: int. The maximum length for the generated name.

    Returns:
        str. A newly generated name.
    """
    # Start with the start token and an empty name
    current_char: str = start_token
    generated_name: str = ""

    # Iterate to build the name
    while current_char != end_token or len(generated_name) <= max_length:
        current_char_index = char_to_idx[current_char]
        next_char = sample_next_character(current_char_index, bigram_probabilities, idx_to_char)
        generated_name += next_char
        current_char = next_char

    return generated_name

def calculate_log_likelihood(
    word: str,
    bigram_probabilities: torch.Tensor,
    char_to_index: Dict[str, int],
    start_token: str = "<S>",
    end_token: str = "<E>",
) -> torch.Tensor:
    """
    Calculate the log likelihood of a word based on bigram probabilities.

    This function computes the log likelihood of a given word using the provided bigram
    probability matrix. The function requires the start and end characters to be specified,
    which are used to process the word for bigram analysis.

    The function iterates through each pair of characters (bigram) in the word, including the
    start and end characters. For each bigram, it looks up the corresponding probability in the
    bigram probability matrix and computes the log of this probability. The log likelihood of the
    word is the sum of these log probabilities.

    Args:
        word: str. The word for which to calculate the log likelihood.
        bigram_probabilities: torch.Tensor. A 2D tensor representing the probability of each bigram.
        char_to_index: dict. A dictionary mapping characters to their indices in the probability matrix.
        start_char: str. The character that denotes the start of a word. Shall be a single character.
        end_char: str. The character that denotes the end of a word. Shall be a single character.

    Returns:
        Tensor. The log likelihood of the word.
    """
    # Add start and end characters to the word
    processed_word: str = start_token + word.lower() + end_token

    # Initialize log likelihood
    log_likelihood: torch.tensor = torch.tensor(0.0, dtype=torch.float32)

    # Iterate through bigrams in the word and accumulate their log probabilities
    
    for i in range(len(processed_word)-1):
        char1 = processed_word[i]
        char2 = processed_word[i+1]
        if char1 not in char_to_index or char2 not in char_to_index:
            print(f"Missing bigram: {char1}, {char2}. Skipping.")
            continue
        bigram_index = (char_to_index[char1], char_to_index[char2])
        P_bigram = bigram_probabilities[bigram_index]
        if P_bigram <= 0:
            P_bigram = 1e-10 
        log_likelihood += torch.log(P_bigram)

    return log_likelihood


if __name__ == "__main__":
    pass
