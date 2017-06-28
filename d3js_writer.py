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
import json
import os

from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec


class D3GeneratorBase(object):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = float(threshold)
        self.output = {'nodes': [],
                       'links': [],
                       'threshold': self.threshold}

        # doctags is a dict of labels and DocTag objects; we need to iterate
        # over something with a stable order, because otherwise the notion of
        # 'subsequent' discussed in the docstring has no meaning.
        self.labels = list(model.docvecs.doctags.keys())

    def _finish(self, filename=None):
        if not filename:
            filename = 'datavis/datafiles/output.json'
        else:
            filename = 'datavis/datafiles/' + filename + '.json'

        with open(filename, 'w') as f:
            f.write(json.dumps(self.output))

        print('Done!')


class D3Generator(D3GeneratorBase):
    """This generates D3 files for document networks."""

    def __init__(self, model, threshold):
        super(D3Generator, self).__init__(model, threshold)
        self.DOCS_RELATIVE_DIR = 'documents'

    def _find_xml_for(self, label):
        # Look through subdirectories of the documents folder to find an
        # xml file corresponding to this label, and return the relative
        # filepath if we find one.
        for root, dir, files in os.walk(self.DOCS_RELATIVE_DIR):
            for file in files:
                if '-new.txt' in label:
                    xml_label = label.replace('-new.txt', '.xml')
                else:
                    xml_label = label.replace('.txt', '.xml')

                if file == xml_label:
                    return os.path.join(root, file)

        # We didn't find anything, so return None and let the caller figure out
        # what to do.
        return None

    def _parse_xml(self, xml):
        title, author, advisor, dlc, url = None, None, None, None, None

        with open(xml) as soupfile:
            soup = BeautifulSoup(soupfile, 'xml')

            # Get author, title, dlc
            entities = soup.find_all('roleTerm')
            for entity in entities:
                if entity.string == 'advisor':
                    advisor = entity.find_parent(
                        'role').find_next_sibling('namePart').string
                elif entity.string == 'author':
                    author = entity.find_parent(
                        'role').find_next_sibling('namePart').string
                elif entity.string == 'other':
                    dlc = entity.find_parent(
                        'role').find_next_sibling('namePart').string

            # Get title
            try:
                title = soup.find('title').string
            except AttributeError:
                # If we don't find a title element, an AttributeError will be
                # thrown when we try to access string.
                pass

            # Get URL
            try:
                id = soup.find('identifier')
                assert id.attrs['type'] == 'uri'
                url = id.attrs['type'].string
            except (AttributeError, AssertionError):
                pass

        return title, author, advisor, dlc, url

    def _write_node(self, label):
        """The format of an individual node is
            {"id": "Myriel", "other": 1, "data": 2, "as": 3, "needed": 4}"""

        # Check for XML file with metadata for this document
        xml = self._find_xml_for(label)
        title, author, advisor, dlc, url = self._parse_xml(xml)

        # If it exists, add a node with useful metadata
        if xml:
            self.output['nodes'].append({'id': label,
                                         'title': title,
                                         'author': author,
                                         'advisor': advisor,
                                         'dlc': dlc,
                                         'url': url})

        # If it doesn't, add a node with the (scant) available data
        else:
            self.output['nodes'].append({'id': label})

    def _write_links(self, label, index):
        """The format of an individual link is
            {"source": "Napoleon", "target": "Myriel", "value": 1}"""

        i = index + 1
        while i < len(self.labels):
            target = self.labels[i]
            line = {'source': label,
                    'target': target,
                    'value': self.model.docvecs.similarity(label, target)}
            if line['value'] > self.threshold:
                self.output['links'].append(line)
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
            line = {'source': word,
                    'target': target[0],
                    'value': target[1]}
            if line['value'] > self.threshold:
                self.output['links'].append(line)
            if current_hop < hops:
                self._inner_make_word_network(target[0], hops, current_hop + 1)

    def _write_node(self, label):
        # This shouldn't be on D3GeneratorBase because D3Generator will have a
        # very different write_node once we start harvesting XML metadata.
        """The format of an individual node is
            {"id": "Myriel", "other": 1, "data": 2, "as": 3, "needed": 4}"""
        self.output['nodes'].append({'id': label})

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
    parser.add_argument('-t', '--threshold', default=0.5, help="Minimum threshold value for link to be included in visualization file.")
    args = parser.parse_args()
    model = Doc2Vec.load(args.filename)

    if args.word:
        generator = D3GeneratorWords(model, args.threshold)
        hops = args.hops if args.hops else 3
        generator.execute(args.word, hops)
    else:
        generator = D3Generator(model, args.threshold)
        generator.execute()
