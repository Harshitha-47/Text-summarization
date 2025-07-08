import nltk
# Uncomment the next line if running for the first time to download stopwords
# nltk.download('stopwords')

from nltk.corpus import stopwords  # Import English stopwords
from nltk.cluster.util import cosine_distance  # Used to calculate cosine similarity
import numpy as np  # Numerical operations and matrices
import networkx as nx  # Used to build the graph and apply PageRank
import re  # For cleaning the text

# Function to read text from file and convert to cleaned sentence-word lists
def read_article(file_name):
    file = open(file_name, "r")  # Open the file
    filedata = file.readlines()  # Read all lines
    article = filedata[0].split(".")  # Split text into sentences using period
    sentences = []
    for sentence in article:
        cleaned_sentence = re.sub("[^a-zA-Z]", " ", sentence)  # Remove non-letter characters
        sentences.append(cleaned_sentence.split())  # Convert to list of words
    sentences.pop()  # Remove last empty sentence if any
    return sentences

# Function to calculate similarity between two sentences using cosine distance
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    # Convert to lowercase
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    # Unique set of words from both sentences
    all_words = list(set(sent1 + sent2))

    # Create frequency vectors
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    # If both vectors are zero, return 0 similarity
    if np.all((np.array(vector1) == 0)) or np.all((np.array(vector2) == 0)):
        return 0

    # Return similarity score
    return 1 - cosine_distance(vector1, vector2)

# Function to create the similarity matrix for all sentence pairs
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))  # Initialize zero matrix
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # Skip same sentence
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(
                sentences[idx1], sentences[idx2], stop_words
            )
    return similarity_matrix

# Main function to generate and print the summary
def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')  # Load English stopwords
    summarize_text = []  # To store selected sentences

    sentences = read_article(file_name)  # Read and clean the text
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)  # Build similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)  # Create graph from matrix
    scores = nx.pagerank(sentence_similarity_graph)  # Apply PageRank to rank sentences

    # Sort sentences based on PageRank score
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Take top 'n' sentences
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))  # Convert word list to sentence

    # Print the summary
    print("Summary:\n", ". ".join(summarize_text))

# Call the function with your file name and number of summary lines
generate_summary("msft.txt", 2)
