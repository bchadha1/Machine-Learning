from argparse import ArgumentParser
import sys
import os
import re
from operator import mul
from functools import reduce


word_regex = re.compile('[a-zA-Z\']+(?:-[a-zA-Z\']+)?')

def Open_Emails(folder):
    Emails = {}
    filenames = list(os.walk(folder))[0][2]

    # Step through all files in folder
    for filename in filenames:
        file_path = folder + "/" + filename

        # Read file, ignoring invalid
        with open(file_path, encoding=sys.stdout.encoding, errors="ignore") as f:
            # Add message to dict of Emails
            Emails[filename] = f.read()
    return Emails

def get_word_frequencies(words):
    word_frequencies = {}
    total_messages = len(words)

    # Count up occurences of each word in Emails
    for key, message in words.items():
        for word in message:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    for word, occurences in word_frequencies.items():
        word_frequencies[word] = (occurences / total_messages, occurences)

    return word_frequencies

def word_occurences(Emails, phrase_length):
    occurences_of_word = {}

    for k, email in Emails.items():
        words = word_regex.findall(email)
        num_words = len(words)
        words_of_email = {}

        for length in range(phrase_length):
            for start in range(num_words - length):
                phrase = " ".join(words[start + i] for i in range(length + 1))

                if phrase not in words_of_email:
                    words_of_email[phrase] = 1
                else:
                    words_of_email[phrase] = words_of_email[phrase] + 1

        occurences_of_word[k] = words_of_email
    return occurences_of_word

# Return the spam score of a message
def get_spam_score(message, word_spamicities):
    # Ignore words that have not been encountered
    message = [word for word in message if word in word_spamicities]

    spamicities = [(word, word_spamicities[word]) for word in message]

    prob_spam = reduce(mul, [x[1] for x in sorted(spamicities, key=lambda x: abs(0.5 - x[1]), reverse=True)[:10]])
    prob_spam_inv = reduce(mul, [1 - x[1] for x in sorted(spamicities, key=lambda x: abs(0.5 - x[1]), reverse=True)[:10]])

    probability_of_spam_plus_spam_inv = prob_spam + prob_spam_inv
    return prob_spam / probability_of_spam_plus_spam_inv

def get_word_spamicities(spam_word_frequencies, ham_word_frequencies, init_prob_spam, occurence_threshold):
    spamicities = {}
    words = set(list(spam_word_frequencies) + list(ham_word_frequencies))

    for word in words:
        spam_word_occurences = spam_word_frequencies[word][1]\
            if word in spam_word_frequencies else 0
        ham_word_occurences = ham_word_frequencies[word][1]\
            if word in ham_word_frequencies else 0

        # Do not include word if it occurs less times than threshold in total
        spam_plus_ham_word_occurence = spam_word_occurences + ham_word_occurences
        if spam_plus_ham_word_occurence >= occurence_threshold:
            if word in spam_word_frequencies and word in ham_word_frequencies:

                spam_word_probability = spam_word_frequencies[word][0] * init_prob_spam
                ham_word_probability = ham_word_frequencies[word][0] * (1 - init_prob_spam)

                spam_plus_ham_word_prob = spam_word_probability + ham_word_probability
                spamicities[word] = spam_word_probability / spam_plus_ham_word_prob
            # Word is not present in spam Emails
            elif spam_word_occurences == 0:
                spamicities[word] = 0.01
            # Word is not present in ham Emails
            elif ham_word_occurences == 0:
                spamicities[word] = 0.99

    return spamicities

parser = ArgumentParser(description="Naive Bayes Spam Filter")
parser.add_argument("train_spam")
parser.add_argument("train_ham")
parser.add_argument("test_spam")
parser.add_argument("test_ham")
parser.add_argument("-init_prob_spam", default="0.5", type=float)
parser.add_argument("-occurence_threshold", default="10", type=int)
parser.add_argument("-score_threshold",  default="0.9", type=float)
parser.add_argument("-phrase_length", default="1", type=int)
args = parser.parse_args()

# Get initial probability of spam
init_prob_spam = args.init_prob_spam

# Get thresholds
occurence_threshold = args.occurence_threshold
score_threshold = args.score_threshold

# Get Emails from files
spam_example_messages = Open_Emails(args.train_spam)
ham_example_messages = Open_Emails(args.train_ham)
spam_test_messages = Open_Emails(args.test_spam)
ham_test_messages = Open_Emails(args.test_ham)

# Get spam and ham word occurences per message
spam_words = word_occurences(spam_example_messages, args.phrase_length)
ham_words = word_occurences(ham_example_messages, args.phrase_length)
spam_test_words = word_occurences(spam_test_messages, args.phrase_length)
ham_test_words = word_occurences(ham_test_messages, args.phrase_length)

spam_word_frequencies = get_word_frequencies(spam_words)
ham_word_frequencies = get_word_frequencies(ham_words)

word_spamicities = get_word_spamicities(spam_word_frequencies, ham_word_frequencies, init_prob_spam, occurence_threshold)

spam_successes = 0
ham_successes = 0

for key in sorted(spam_test_words):
    message_score = get_spam_score(spam_test_words[key], word_spamicities)
    # print(key, ":", message_score)
    if message_score > score_threshold:
        spam_successes += 1

for key in sorted(ham_test_words):
    message_score = get_spam_score(ham_test_words[key], word_spamicities)
    # print(key, ":", message_score)
    if message_score < score_threshold:
        ham_successes = ham_successes + 1
        ham_successes = ham_successes + 1


test_ham_len = len(ham_test_messages)
ham_success_rate = ham_successes / test_ham_len

test_spam_len = len(spam_test_messages)
spam_success_rate = spam_successes / test_spam_len

print("Spam Successes: ", spam_successes)
print("Spam success rate: {0:.2f}%".format(spam_success_rate * 100))
print("")
print("Ham Successes: ", ham_successes)
print("Ham success rate: {0:.2f}%".format(ham_success_rate * 100))
