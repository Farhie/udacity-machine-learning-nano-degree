import pandas

sms_dataset = pandas.read_table(filepath_or_buffer='./SMSSpamCollection', sep='\t', names=['label', 'sms_message'])

sms_dataset.label = sms_dataset.label.map({'ham': 0, 'spam': 1})

print(sms_dataset.head(5))
