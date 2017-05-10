import argparse
import os
import time

from gensim.models.doc2vec import LabeledSentence, Doc2Vec

# Get documents
# Thoughts...
# It's 843G of docs. I only have 402 on my machine. So I need to plan on
# starting with a subset - which I should *anyway* - but I also need to think
# about what's resident in memory when.
# Do I need to load the entire corpus into memory to do training or is a
# stepwise thing happening?
# For step 1, I should just get a single-department subdirectory and scp it
# over to my machine. The goal here is to make sure the code runs at all,
# address tokenization, etc.
# Next step will involve network file access.

# See https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_RELATIVE_DIR = 'documents'
DOCS_ABSOLUTE_DIR = os.path.join(BASE_DIR, DOCS_RELATIVE_DIR)


class LabeledLineSentence(object):
    def __init__(self, doc_list):
        self.doc_list = doc_list

    def __iter__(self):
        for doc in self.doc_list:
            yield LabeledSentence(words=self._prep_document(doc), tags=[doc])

    def _prep_document(self, doc):
        """Given a document filename, opens the file and tokenizes it."""
        full_path = os.path.join(DOCS_ABSOLUTE_DIR, doc)
        with open(full_path, 'r') as doc_contents:
            return doc_contents.read().split()


def get_iterator():
    doc_list = [f for f in os.listdir(DOCS_ABSOLUTE_DIR)
                if os.path.splitext(f)[-1] == '.txt']
    return LabeledLineSentence(doc_list)


def train_model(filename):
    model = Doc2Vec(alpha=0.025,
                    min_alpha=0.025)

    doc_iterator = get_iterator()
    model.build_vocab(doc_iterator)

    for epoch in range(10):
        start_time = time.time()
        print("=== Training epoch {} ===".format(epoch))
        model.train(doc_iterator,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        print("Finished training, took {}".format(time.time() - start_time))

    model.save('{filename}.model'.format(filename=filename))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural net on .txt '
        'files located in the documents/ directory.')
    parser.add_argument('filename')

    args = parser.parse_args()
    train_model(args.filename)
