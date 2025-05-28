import numpy as np
from symrank import cosine_similarity

def main():
    # Generate a random query vector
    query_vector = np.random.rand(1536).astype(np.float32)

    # Generate random candidate vectors
    candidate_vectors = [
        (f"doc_{i}", np.random.rand(1536).astype(np.float32))
        for i in range(300)
    ]

    # Call the compare function
    print("Calling compare function...")
    top_results = cosine_similarity(query_vector, candidate_vectors)

    # Print the results
    print("Top Results:")
    for result in top_results:
        print(result)

if __name__ == "__main__":
    main()