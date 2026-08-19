#compares new content(embeddings) with creator profile (embeddings) to check for similarity and returns a similarity score

def check_similarity(new_content_embedding, creator_profile_embedding):
    # Calculate the dot product of the two embeddings
    dot_product = sum(a * b for a, b in zip(new_content_embedding, creator_profile_embedding))
    
    # Calculate the magnitude of each embedding
    magnitude_new = sum(a ** 2 for a in new_content_embedding) ** 0.5
    magnitude_creator = sum(b ** 2 for b in creator_profile_embedding) ** 0.5
    
    # Calculate the cosine similarity
    if magnitude_new == 0 or magnitude_creator == 0:
        return 0.0  # Avoid division by zero; return 0 similarity if either embedding is zero vector
    
    similarity_score = dot_product / (magnitude_new * magnitude_creator)
    
    return similarity_score

