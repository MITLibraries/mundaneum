# README.md

## Installation
This has been tested against python 3.6.1.

Install `requirements.txt`.

Confirm that you have the fast version of gensim:
```
$ python
>>> from gensim.models.doc2vec import FAST_VERSION
>>> FAST_VERSION
```
This should be > -1. If it isn't, try uninstalling and reinstalling gensim (it needs to install after Cython to build the fast version).

In your python shell, `nltk.download('punkt')`.
