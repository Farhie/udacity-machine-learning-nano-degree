import string
from collections import Counter

import pandas
from sklearn.feature_extraction.text import CountVectorizer

sms_dataset = pandas.read_table(filepath_or_buffer='./SMSSpamCollection', sep='\t', names=['label', 'sms_message'])

sms_dataset.label = sms_dataset.label.map({'ham': 0, 'spam': 1})

print("Total number of columns %d" % sms_dataset.shape[1])
print("Total number of rows %d" % sms_dataset.shape[0])
print(sms_dataset.head(5))

# Implementing bag of words from scratch

lowercase_sms_message_series = [sms_message.lower() for sms_message in sms_dataset.sms_message]
punctuation_removed_from_messages = [sms_message.translate(str.maketrans('', '', string.punctuation))
                                     for sms_message in lowercase_sms_message_series]

preprocessed_messages = [sms_message.split(sep=' ') for sms_message in punctuation_removed_from_messages]

frequency_distribution_dictionary = [Counter(sms_message) for sms_message in preprocessed_messages]

print('Output of Bag of Words own implementation: ', frequency_distribution_dictionary)


# Using Sickit learn to implement bag of words

count_vectoriser = CountVectorizer()
count_vectoriser.fit(sms_dataset.sms_message)

print(count_vectoriser.get_feature_names())


