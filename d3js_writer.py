"""
This script takes our corpus and generates a data file usable by d3.js.

Specifically, it...
* Iterates through doctags
* For each doctag:
  * Writes its ID and bibliographic data to 'nodes'
  * Writes its similarity to all SUBSEQUENT doctags into 'links'

Similarity is symmetric, so once we've written the similarity of DocA to DocB,
there is no need to write the similarity of DocB to DocA. This lets us use
n(n-1)/2 rather than n^2 lookups to write the links section (which is still
gigantic, and they're both O(n^2), mind you).

Because this file is large, we don't want to keep it all in memory at the same
time. Therefore we will have two files - one for nodes and one for links -
which we can append to, and we'll merge them at the end as valid JSON.
"""
import argparse
import os
from string import Template

from gensim.models.doc2vec import Doc2Vec


class D3GeneratorBase(object):
    def __init__(self, model):
        self.node_filename = 'nodes.txt'
        self.node_file = open(self.node_filename, 'a')
        self.links_filename = 'links.txt'
        self.links_file = open(self.links_filename, 'a')
        self.model = model
        self.base_link = Template('{"source": $source, "target": $target, '
                                  '"value": $value},')

        # doctags is a dict of labels and DocTag objects; we need to iterate
        # over something with a stable order, because otherwise the notion of
        # 'subsequent' discussed in the docstring has no meaning.
        self.labels = list(model.docvecs.doctags.keys())

    def _cleanup(self):
        os.remove(self.node_filename)
        os.remove(self.links_filename)

    def _close(self):
        self.node_file.close()
        self.links_file.close()

    def _collate_json(self, filename):
        with open(filename, 'w') as f:
            f.write('{"nodes": [')
            with open(self.node_filename) as infile:
                for line in infile:
                    f.write(line)

            f.write('], "links": [')
            with open(self.links_filename) as infile:
                for line in infile:
                    f.write(line)

            f.write(']}')

    def _finish(self, filename=None):
        if not filename:
            filename = 'output.json'
        else:
            filename = filename + '.json'

        self._close()
        self._collate_json(filename=filename)
        self._cleanup()
        print('Done!')


class D3Generator(D3GeneratorBase):
    """This generates D3 files for document networks."""
    def _write_node(self, label):
        """The format of an individual node is
            {"id": "Myriel", "other": 1, "data": 2, "as": 3, "needed": 4}"""
        line = Template('{"id": $label},')
        self.node_file.write(line.substitute(label=label))

    def _write_links(self, label, index):
        """The format of an individual node is
            {"source": "Napoleon", "target": "Myriel", "value": 1}"""

        i = index + 1
        while i < len(self.labels):
            target = self.labels[i]
            line = self.base_link.substitute(source=label,
                target=target,
                value=self.model.docvecs.similarity(label, target))
            self.links_file.write(line)
            i += 1

    def execute(self):
        """Given a Doc2Vec model, manages the overall process of writing the
        json representing the entire model network."""

        for index, label in enumerate(self.labels):
            print('Processing {label}...'.format(label=label))
            self._write_node(label)
            self._write_links(label, index)

        self._finish()


class D3GeneratorWords(D3GeneratorBase):
    """This generates D3 files for the network around a particular word."""
    def _inner_make_word_network(self, word, hops, current_hop=0):
        self._write_node(word)
        for target in self.model.wv.most_similar(word):
            self._write_node(target[0])
            line = self.base_link.substitute(source=word,
                target=target[0],
                value=target[1])
            self.links_file.write(line)
            if current_hop < hops:
                self._inner_make_word_network(target[0], hops, current_hop + 1)

    def _write_node(self, label):
        # This shouldn't be on D3GeneratorBase because D3Generator will have a
        # very different write_node once we start harvesting XML metadata.
        """The format of an individual node is
            {"id": "Myriel", "other": 1, "data": 2, "as": 3, "needed": 4}"""
        line = Template('{"id": $label},')
        self.node_file.write(line.substitute(label=label))

    def execute(self, word, hops):
        self._inner_make_word_network(word, hops)
        self._finish(word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a file usable by '
        'd3.js from a given saved Doc2Vec model.', add_help=False)
    parser.add_argument('filename', help="Base filename of saved model")
    parser.add_argument('-w', '--word', help="Create a network around the "
        "word specified")
    parser.add_argument('-h', '--hops', help="If using -w, how many hops out "
        "from the specified word to traverse. Defaults to 3.")
    args = parser.parse_args()
    model = Doc2Vec.load(args.filename)

    if args.word:
        generator = D3GeneratorWords(model)
        hops = args.hops if args.hops else 3
        generator.execute(args.word, hops)
    else:
        generator = D3Generator(model)
        generator.execute()
