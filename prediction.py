from keras.preprocessing import text, sequence
from keras.models import load_model
import pandas as pd

import string
import re
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

stopwords = ['a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']

APPO = {"aren't" : "are not","can't" : "cannot","couldn't" : "could not","didn't" : "did not","doesn't" : "does not","don't" : "do not","hadn't" : "had not","hasn't" : "has not","haven't" : "have not","he'd" : "he would","he'll" : "he will","he's" : "he is","i'd" : "I would","i'd" : "I had","i'll" : "I will","i'm" : "I am","isn't" : "is not","it's" : "it is","it'll":"it will","i've" : "I have","let's" : "let us","mightn't" : "might not","mustn't" : "must not","shan't" : "shall not","she'd" : "she would","she'll" : "she will","she's" : "she is","shouldn't" : "should not","that's" : "that is","there's" : "there is","they'd" : "they would","they'll" : "they will","they're" : "they are","they've" : "they have","we'd" : "we would","we're" : "we are","weren't" : "were not","we've" : "we have","what'll" : "what will","what're" : "what are","what's" : "what is","what've" : "what have","where's" : "where is","who'd" : "who would","who'll" : "who will","who're" : "who are","who's" : "who is","who've" : "who have","won't" : "will not","wouldn't" : "would not","you'd" : "you would","you'll" : "you will","you're" : "you are","you've" : "you have","'re": " are","wasn't": "was not","we'll":" will","didn't": "did not","tryin'":"trying"}

def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    # Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    # remove \n
    comment=re.sub("\\n"," ",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    # removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    # Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    words=[APPO[word] if word in APPO else word for word in words]
    words = [w for w in words if not w in stopwords]
    
    clean_sent = " ".join(words)
    return clean_sent

def clean_with_stops(comment):
    """
    This function receives comments and returns clean word-list without removing stopwords
    """
    # Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    # remove \n
    comment=re.sub("\\n"," ",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    # removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    # Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    # (')aphostophe  replacement (ie)   you're --> you are  
    words=[APPO[word] if word in APPO else word for word in words]
    #words = [w for w in words if not w in stopwords]
    
    clean_sent = " ".join(words)
    return clean_sent

#test = pd.read_csv('data/test.csv')

#test = test.apply(lambda x: clean(x))

#test.fillna(' ', inplace=True)

#test = test.str.lower()

test = pd.read_csv('data/clean_test.csv')
test['comment_text'].fillna(' ', inplace=True)
test = test['comment_text'].str.lower()
train = pd.read_csv('data/clean_train.csv')
train['comment_text'].fillna(' ', inplace=True)
train_x = train['comment_text'].str.lower()

# Vectorize text + Embedding
tokenizer = text.Tokenizer(num_words=100000, lower=True)
tokenizer.fit_on_texts(train_x.values)

test = tokenizer.texts_to_sequences(test)

test = sequence.pad_sequences(test, maxlen=150)

model = load_model('lstm_fasttext_epoch_5_150d.h5')

predictions = model.predict(test, batch_size=32, verbose=1)

output = pd.read_csv('data/sample_submission.csv')
output[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = predictions
output.to_csv('output.csv', index=False)