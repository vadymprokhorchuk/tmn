import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import csv

from gensim.corpora import Dictionary
from gensim.models import LdaModel

reader = open("book.txt", "r")

# Read file
text = reader.read()

# Find comments at start
start_pos = text.find("CHAPTER I.", text.find("CHAPTER I.") + 1)

# Find comments at the end
end_pos = text.find("THE END")

# Extract the text of book, without comments
text = text[start_pos:end_pos + len("THE END")].strip()

# Download punktuacia database
nltk.download('punkt')

# Download stop words
nltk.download('stopwords')

# Get stop words from list
stop_words = set(stopwords.words('english'))

print("Stop words:\n")
print(stop_words)
print("\n")

# All text to lowercase
text = text.lower()

# Delete numbers
text = re.sub(r'\d+', '', text)

# Delete punctuation marks and non-alphabetic characters (also numbers)
text = re.sub(r'[^a-zA-Z\s]', '', text)

# Splits sentence into words using NLTK
tokens = nltk.word_tokenize(text, language = "english")

# Deleting stop words from all text
filtered_words = [word for word in tokens if word not in stop_words]

# Concat all words array to string
filtered_text = ' '.join(filtered_words)

# Split text into chapters
chapters = re.split(r'chapter ', filtered_text)[1:]

def write_to_csv(name, array):
    with open(name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')

        # Revert
        transposed_data = list(map(list, zip(*array)))

        # Write headers
        headers = ['Chapter ' + str(i + 1) for i in range(len(array))]
        headers.insert(0, '')

        csv_writer.writerow(headers)

        # Записываем транспонированные данные в файл
        for i, row in enumerate(transposed_data):
            csv_writer.writerow([str(i + 1)] + row)

def process_idf():
    print("TF-IDF:")

    # Init lib
    vect = TfidfVectorizer()

    tf_idf_csv_array = []
    # Loop for every chapter
    for index, chapter in enumerate(chapters):
        print(f"Chapter {index + 1}:")

        # Fit data into a model and transform it into a matrix
        matrix = vect.fit_transform([chapter])

        # Getting formatted names from matrix
        names = vect.get_feature_names_out()

        # Sort matrix to get all words, which included in text more than 20 times, anc convert to array
        all_words = [names[i] for i in  matrix.sum(axis=0).argsort()[0, ::-1][:20]]

        # Get first 20 most common used words in chapter
        print(all_words[0][0][:20])
        tf_idf_csv_array.append(all_words[0][0][:20])

    # Write results of TF-IDF to csv file
    write_to_csv('TF-IDF.csv', tf_idf_csv_array)

def process_lda():
    print("LDA:")

    lda_csv_array = []

    for index, chapter in enumerate(chapters):
        print(f"Chapter {index + 1}:")
        chapter_tokens = nltk.word_tokenize(chapter, language = "english")

        # Gensim dictionary to map words to unique ids
        dictionary = Dictionary([chapter_tokens])

        # Converts words aray to list of tuples, contains word's ID and its citation frequency
        corp = [dictionary.doc2bow(token) for token in [chapter_tokens]]

        # Create LDA model with and sort it to one topic by dictionary
        model = LdaModel(corp, num_topics=1, id2word=dictionary)

        # Extracting the topic with 20 top words in this topic
        topics = model.show_topics(num_topics=1, num_words=20, formatted=False)

        # Extract words from topic
        top_words = [word[0] for word in topics[0][1]]

        print(top_words)

        lda_csv_array.append(top_words)

    write_to_csv('LDA.csv', lda_csv_array)
        

process_idf()
process_lda()