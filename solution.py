import json
import pathlib
import gensim
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import string


class ChatBot:
    name = 'Miracle'
    corpus_path = 'jeopardy.json'
    model_path = 'my_trained_model'

    def __init__(self):
        self.corpus_file = None
        self.corpus = None
        self.main()

    def main(self):
        self.greeting()
        self.corpus = self.read_corpus()
        self.train_model()

    def train_model(self):
        if not pathlib.Path(self.model_path).is_file():
            my_model = gensim.models.doc2vec.Doc2Vec(vector_size=200, window=4, min_count=5, workers=4, epochs=30)
            my_model.build_vocab(self.corpus)
            my_model.train(self.corpus, total_examples=my_model.corpus_count, epochs=my_model.epochs)
            my_model.save(self.model_path)
        else:
            my_model = gensim.models.doc2vec.Doc2Vec.load(self.model_path)

        self.ask_model(my_model)

    def ask_model(self, model):
        while True:
            print("Ask me something!")
            request = self.get_user_input().lower()
            tokenize_request = self.preprocess_text(request)

            query_vector = model.infer_vector(tokenize_request)
            most_similar_doc = model.dv.most_similar([query_vector], topn=1)
            most_similar_topic_index, similarity_percentage = most_similar_doc[0]
            answer = self.corpus_file[int(most_similar_topic_index)]['answer']
            print(f"I know this question: its number is {most_similar_topic_index}. "
                  f"I'm {int(similarity_percentage * 100)}% sure of this.")
            print(f"The answer is {answer}")

            print("Do you want to ask me again? (yes/no)")
            choice = input().lower()
            if choice == 'no':
                print('It was nice to play with you! Goodbye!')
                break

    def greeting(self):
        print(f"Hello! I'm {self.name}, a question answering bot who knows answers to all questions from the "
              f"'Jeopardy!' game.")

    @staticmethod
    def get_user_input():
        user_input = input()
        return user_input

    def read_corpus(self):
        corpus_file = json.loads(open(self.corpus_path, "rb").read())
        self.corpus_file = corpus_file
        processed_corpus = []
        for i, sentence in enumerate(corpus_file):
            processed_sentence = self.preprocess_text(sentence['question'])
            tagged_sentence = TaggedDocument(words=processed_sentence, tags=[str(i)])
            processed_corpus.append(tagged_sentence)
        return processed_corpus

    @staticmethod
    def preprocess_text(text):
        text = text.lower()

        tokens = word_tokenize(text)

        tokens = [token for token in tokens if token not in string.punctuation]

        return tokens


chatBot = ChatBot()
