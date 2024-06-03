def normalize_capitalization(sentence):
    """Takes a sentence and normalises capitalisation."""
    return sentence.lower()

def tokenize(sentence):
    """Returns a list of words."""
    return sentence.split()

def remove_stop_words(sentence, stop_words):
    """Takes a sentence and removes stop words."""
    normalized_sentence = tokenize(sentence)
    pruned_sentence = []
    stop_words_normalized = tokenize(stop_words)
    for word in normalized_sentence:
        if word not in stop_words_normalized:
            pruned_sentence.append(word)
    return " ".join(pruned_sentence)

def remove_punctuation(sentence, punctuation):
    """Returns a string without trailing punctuation."""
    words_sentence = tokenize(sentence)
    pruned_sentence = []
    for word in words_sentence:
        while word and word[-1] in punctuation:
            word = word[:-1]
        pruned_sentence.append(word)
    return " ".join(pruned_sentence)

def remove_duplicates(sentence):
    """Removes duplicate words."""
    word_sentence = tokenize(sentence)
    unique_words = []
    pruned_sentence = []
    for word in word_sentence:
        if word not in unique_words:
            pruned_sentence.append(word)
            unique_words.append(word)
    return " ".join(sorted(pruned_sentence))

def clean_noise(sentence):
    """Removes noise."""
    normalized_sentence = tokenize(sentence.replace("&amp", "&"))
    cleaned_sentence = []
    odd_counter = 0
    for word in normalized_sentence:
        if '@' in word:
            odd_counter += 1
        if 'http' not in word and '#' not in word and '@' not in word:
            cleaned_sentence.append(word)
        if odd_counter % 2 == 0 and "@" in word:
            cleaned_sentence.append(word)
    return " ".join(cleaned_sentence)

def construct_ngrams(sentence, n):
    """Takes a sentence and makes ngrams."""
    ngrams = []
    normalized_sentence = tokenize(sentence)
    if n <= 0 or len(normalized_sentence) < n:
        return ngrams
    for i in range(len(normalized_sentence) - n + 1):
        ngrams.append(normalized_sentence[i:i + n])
    return ngrams

def stem_words(sentence):
    """Converts words to root forms."""
    normalized_sentence = tokenize(sentence)
    exempt_for_s = ["us", "ss", "'s", "as", "es", "is", "os", "ys"]
    forbidden_endings = ["'s", "s'", "ed", "er", "ly"]
    stemmed_sentence = []
    for word in normalized_sentence:
        can_stem = True
        while can_stem:
            can_stem = False
            if word[-1] == "s" and word[-2:] not in exempt_for_s:
                can_stem = True
                word = word[:-1]
            elif word[-4:] == "sses":
                can_stem = True
                word = word[:-2]
            elif word[-3:] in {"ies", "ied"}:
                can_stem = True
                word = word[:-2]
                if len(word) <= 2:
                    word = word[:-1]
            elif word[-3:] == "ing" and len(word) > 5:
                can_stem = True
                word = word[:-3]
            elif word[-2:] in forbidden_endings:
                can_stem = True
                word = word[:-2]
        stemmed_sentence.append(word)
    return " ".join(stemmed_sentence)

def load_data(filename):
    """Takes a filename and returns all the lines."""
    with open(filename, encoding="utf8") as file:
        return [line.strip() for line in file]

def analyze_tweets():
    """Uses functions to analyse tweets."""
    input_file = input("Enter the name of the file to read: ")
    output_file = input("Enter the name of the file to write: ")
    stop_words = input("Enter your stopwords: ")
    punctuation = input("Enter your punctuation to remove: ")

    data = load_data(input_file)
    pruned_list = []
    for line in data:
        print(line)
        line = normalize_capitalization(line)
        line = remove_stop_words(line, stop_words)
        line = remove_punctuation(line, punctuation)
        line = remove_duplicates(line)
        line = clean_noise(line)
        line = stem_words(line)
        with open(output_file, "w+", encoding="utf8") as f:
            f.write(line)
        pruned_list.append(line)
    return pruned_list

def rank_words(corpus, n):
    """Takes a list of words and ranks the top n."""
    sorted_normalized_words = sorted(tokenize(" ".join(corpus)))
    unique_words = []
    words_with_count = []
    counts = []
    top_n_freq = []
    final_words = []

    current_word = sorted_normalized_words[0]
    for word in sorted_normalized_words:
        if word != current_word and word not in unique_words:
            unique_words.append(word)
            current_word = word

    sorted_unique_words = sorted(unique_words)
    current_word_freq = sorted_unique_words[0]
    count = 0
    for word in sorted_normalized_words:
        if word == current_word_freq:
            count += 1
        else:
            words_with_count.append((current_word_freq, count))
            count = 1
            current_word_freq = word
    words_with_count.append((current_word_freq, count))

    for word, count in words_with_count:
        counts.append(count)

    for _ in range(n):
        max_count = max(counts)
        counts.remove(max_count)
        top_n_freq.append(max_count)

    for word, count in words_with_count:
        if count in top_n_freq:
            final_words.append((word, count))
    return final_words
