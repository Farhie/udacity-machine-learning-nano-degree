import pandas

sms_dataset = pandas.read_table(filepath_or_buffer='./SMSSpamCollection', sep='\t', names=['label', 'sms_message'])

print(sms_dataset.head(5))
