## Our Mission ##

Spam detection is one of the major applications of Machine Learning in the interwebs today. Pretty much all of the
major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'.

In this mission we will be using the Naive Bayes algorithm to create a model that can classify
'https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection' SMS messages as spam or not spam,
based on the training we give to the model. It is important to have some level of intuition as to what a spammy
text message might look like. Usually they have words like 'free', 'win', 'winner', 'cash', 'prize' and the like in
them as these texts are designed to catch your eye and in some sense tempt you to open them. Also, spam messages tend
to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually
pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!

Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or
'Not Spam' and nothing else. Also, this is a supervised learning problem, as we will be feeding a labelled dataset
into the model, that it can learn from, to make future predictions.


### Step 0: Introduction to the Naive Bayes Theorem ###
Bayes theorem is one of the earliest probabilistic inference algorithms developed by Reverend Bayes (which he used to try and infer the existence of God no less) and still performs extremely well for certain use cases.

It's best to understand this theorem using an example. Let's say you are a member of the Secret Service and you have been deployed to protect the Democratic presidential nominee during one of his/her campaign speeches. Being a public event that is open to all, your job is not easy and you have to be on the constant lookout for threats. So one place to start is to put a certain threat-factor for each person. So based on the features of an individual, like the age, sex, and other smaller factors like is the person carrying a bag?, does the person look nervous? etc. you can make a judgement call as to if that person is viable threat.

If an individual ticks all the boxes up to a level where it crosses a threshold of doubt in your mind, you can take action and remove that person from the vicinity. The Bayes theorem works in the same way as we are computing the probability of an event(a person being a threat) based on the probabilities of certain related events(age, sex, presence of bag or not, nervousness etc. of the person).

One thing to consider is the independence of these features amongst each other. For example if a child looks nervous at the event then the likelihood of that person being a threat is not as much as say if it was a grown man who was nervous. To break this down a bit further, here there are two features we are considering, age AND nervousness. Say we look at these features individually, we could design a model that flags ALL persons that are nervous as potential threats. However, it is likely that we will have a lot of false positives as there is a strong chance that minors present at the event will be nervous. Hence by considering the age of a person along with the 'nervousness' feature we would definitely get a more accurate result as to who are potential threats and who aren't.

This is the 'Naive' bit of the theorem where it considers each feature to be independant of each other which may not always be the case and hence that can affect the final judgement.

In short, the Bayes theorem calculates the probability of a certain event happening(in our case, a message being spam) based on the joint probabilistic distributions of certain other events(in our case, a message being classified as spam). We will dive into the workings of the Bayes theorem later in the mission, but first, let us understand the data we are going to work with.

### Step 1.1: Understanding our dataset ###


We will be using a 'https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection' dataset from the UCI Machine Learning repository which has a very good collection of datasets for experimental research purposes.

The columns in the data set are currently not named and as you can see, there are 2 columns.

The first column takes two values, 'ham' which signifies that the message is not spam, and 'spam' which signifies that the message is spam.

The second column is the text content of the SMS message that is being classified.

> Instructions:
* Import the dataset into a pandas dataframe using the read_table method. Because this is a tab separated dataset we
will be using '\t' as the value for the 'sep' argument which specifies this format.
* Also, rename the column names by specifying a list ['label, 'sms_message'] to the 'names' argument of read_table().
* Print the first five values of the dataframe with the new column names.


Step 1.2: Data Preprocessing

Now that we have a basic understanding of what our dataset looks like, lets convert our labels to binary variables, 0 to represent 'ham'(i.e. not spam) and 1 to represent 'spam' for ease of computation.

You might be wondering why do we need to do this step? The answer to this lies in how scikit-learn handles inputs. Scikit-learn only deals with numerical values and hence if we were to leave our label values as strings, scikit-learn would do the conversion internally(more specifically, the string labels will be cast to unknown float values).

Our model would still be able to make predictions if we left our labels as strings but we could have issues later when calculating performance metrics, for example when calculating our precision and recall scores. Hence, to avoid unexpected 'gotchas' later, it is good practice to have our categorical values be fed into our model as integers.


>Instructions:
* Convert the values in the 'label' colum to numerical values using map method as follows:
{'ham':0, 'spam':1} This maps the 'ham' value to 0 and the 'spam' value to 1.
* Also, to get an idea of the size of the dataset we are dealing with, print out number of rows and columns using
'shape'.
