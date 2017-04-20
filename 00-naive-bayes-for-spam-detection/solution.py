import pandas

sms_dataset = pandas.read_table(filepath_or_buffer='./SMSSpamCollection', sep='\t', names=['label', 'sms_message'])

sms_dataset.label = sms_dataset.label.map({'ham': 0, 'spam': 1})

print("Total number of columns %d" % sms_dataset.shape[1])
print("Total number of rows %d" % sms_dataset.shape[0])
print(sms_dataset.head(5))

# Implementing bag of words from scratch

sms_message_series = [sms_message.lower() for sms_message in sms_dataset.sms_message]

print(sms_message_series)
