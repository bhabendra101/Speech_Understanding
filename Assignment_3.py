def words2characters(words):
    """
    This function converts a list of words into a list of characters.
    """
    characters = []

    for word in words:
        # Convert each element to string
        for char in str(word):
            characters.append(char)

    return characters
