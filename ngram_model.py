
from collections import Counter, defaultdict
import random
import pandas as pd
import textwrap

# Load the dataset (as it's used by build_ngram_model)
africa_galore = pd.read_json(
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"
)
dataset = africa_galore["description"]

def space_tokenize(text: str) -> list[str]:
    """Splits a string into a list of words (tokens)."""
    tokens = text.split(" ")
    return tokens

def generate_ngrams(text: str, n: int) -> list[tuple[str]]:
    """Generates n-grams from a given text."""
    tokens = space_tokenize(text)
    ngrams = []
    num_of_tokens = len(tokens)
    for i in range(0, num_of_tokens - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def get_ngram_counts(dataset: list[str], n: int) -> dict[str, Counter]:
    """Computes the n-gram counts from a dataset.

    This function takes a list of text strings (paragraphs or sentences) as
    input, constructs n-grams from each text, and creates a dictionary where:

    * Keys represent n-1 token long contexts `context`.
    * Values are a Counter object `counts` such that `counts[next_token]` is the
      count of `next_token` following `context`.

    Args:
        dataset: The list of text strings in the dataset.
        n: The size of the n-grams to generate (e.g., 2 for bigrams, 3 for
            trigrams).

    Returns:
        A dictionary where keys are (n-1)-token contexts and values are Counter
        objects storing the counts of each next token for that context.

    """
    ngram_counts = defaultdict(Counter)
    for paragraph in dataset:
        for ngram in generate_ngrams(paragraph, n):
            context = " ".join(ngram[:-1])
            next_token = ngram[-1]
            ngram_counts[context][next_token] += 1
    return dict(ngram_counts)

def build_ngram_model(
    dataset: list[str],
    n: int
) -> dict[str, dict[str, float]]:
    """Builds an n-gram language model.

    This function takes a list of text strings (paragraphs or sentences) as
    input, generates n-grams from each text using the function get_ngram_counts
    and converts them into probabilities.  The resulting model is a dictionary,
    where keys are (n-1)-token contexts and values are dictionaries mapping
    possible next tokens to their conditional probabilities given the context.

    Args:
        dataset: A list of text strings representing the dataset.
        n: The size of the n-grams (e.g., 2 for a bigram model).

    Returns:
        A dictionary representing the n-gram language model, where keys are
        (n-1)-tokens contexts and values are dictionaries mapping possible next
        tokens to their conditional probabilities.
    """
    ngram_model = {}
    ngram_counts = get_ngram_counts(dataset, n)
    for context, next_tokens in ngram_counts.items():
        context_total_count = sum(next_tokens.values())
        ngram_model[context] = {}
        for token, count in next_tokens.items():
            ngram_model[context][token] = count / context_total_count
    return ngram_model

def generate_next_n_tokens(
    n: int,
    ngram_model: dict[str, dict[str, float]],
    prompt: str,
    num_tokens_to_generate: int,
) -> str:
    """Generates `num_tokens_to_generate` tokens following a given prompt using
    an n-gram language model.

    This function takes an n-gram model and uses it to predict the most
    likely next token for the given prompt. The generation process
    continues iteratively, appending predicted tokens to the prompt until the
    desired number of tokens is generated or a context is
    encountered for which the model has no predictions.

    Args:
        n: The size of the n-grams to use (e.g., 2 for a bigram model).
        ngram_model: A dictionary representing the n-gram language model.
        prompt: The starting text prompt for generating the next tokens.
        num_tokens_to_generate: The number of words to generate following
            the prompt.

    Returns:
        A string containing the original prompt followed by the generated
        tokens. If no valid continuation is found for a given context, the
        function will return the text generated up to that point and print a
        message indicating that no continuation could be found.
    """
    generated_words = space_tokenize(prompt)
    for _ in range(num_tokens_to_generate):
        context = generated_words[-(n - 1):]
        context = " ".join(context)
        if context in ngram_model:
            next_word = random.choices(
                list(ngram_model[context].keys()),
                weights=ngram_model[context].values()
            )[0]
            generated_words.append(next_word)
        else:
            print(
                "⚠️ No valid continuation found. Change the prompt or"
                " try sampling another continuation.\n"
            )
            break
    return " ".join(generated_words)

# Example Usage:
if __name__ == '__main__':
    print("Building trigram model...")
    trigram_model = build_ngram_model(dataset, n=3)
    print("Model built.")

    prompt = "Jide was hungry so she went looking for"
    n = 3  # Trigram.
    num_tokens_to_generate = 10  # Generate next n words.
    generated_text = generate_next_n_tokens(
        n=n,
        ngram_model=trigram_model,
        prompt=prompt,
        num_tokens_to_generate=num_tokens_to_generate,
    )
    print("\nGenerated text:")
    print(generated_text)
