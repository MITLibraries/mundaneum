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

"""
d3.js generator script needs to...
"""


class D3Generator(object):
    def __init__(self, model):
        self.node_filename = 'nodes.txt'
        self.node_file = open(self.node_filename, 'a')
        self.links_filename = 'links.txt'
        self.links_file = open(self.links_filename, 'a')
        self.model = model

        # doctags is a dict of labels and DocTag objects; we need to iterate
        # over something with a stable order, because otherwise the notion of
        # 'subsequent' discussed in the docstring has no meaning.
        self.labels = list(model.docvecs.doctags.keys())

    def cleanup(self):
        os.remove(self.node_filename)
        os.remove(self.links_filename)

    def close(self):
        self.node_file.close()
        self.links_file.close()

    def collate_json(self):
        with open('output.json', 'w') as f:
            f.write('{"nodes": [')
            with open(self.node_filename) as infile:
                for line in infile:
                    f.write(line)

            f.write('], "links": [')
            with open(self.links_filename) as infile:
                for line in infile:
                    f.write(line)

            f.write(']}')

    def execute(self):
        """Given a Doc2Vec model, manages the overall process of writing the
        json."""

        for index, label in enumerate(self.labels):
            print('Processing {label}...'.format(label=label))
            self.write_node(label)
            self.write_links(label, index)

        self.close()
        self.collate_json()
        self.cleanup()
        print('Done!')

    def write_node(self, label):
        """The format of an individual node is
            {"id": "Myriel", "other": 1, "data": 2, "as": 3, "needed": 4}"""
        line = Template('{"id": $label},')
        self.node_file.write(line.substitute(label=label))

    def write_links(self, label, index):
        """The format of an individual node is
            {"source": "Napoleon", "target": "Myriel", "value": 1}"""

        base_line = Template('{"source": $source, "target": $target, '
                             '"value": $value},')

        i = index + 1
        while i < len(self.labels):
            target = self.labels[i]
            line = base_line.substitute(source=label,
                target=target,
                value=self.model.docvecs.similarity(label, target))
            self.links_file.write(line)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a file usable by '
        'd3.js from a given saved Doc2Vec model.')
    parser.add_argument('filename', help="Base filename of saved model")
    args = parser.parse_args()
    model = Doc2Vec.load(args.filename)

    generator = D3Generator(model)
    generator.execute()
