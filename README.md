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

## Getting files

You need to be inside the MIT network (i.e. via VPN if off-campus), and you need an account on repo-dev-1.mit.edu.

### Using the script
* Set up an ssh keypair and install your public key on repo-dev-1:
  * `ssh keygen -t rsa`
  * `ssh-copy-id <your kerb>@repo-dev-1.mit.edu`
* `python initial-test.py -n <thesis subdir you want to fetch> -k <your kerb>`
  * The subdir will be assumed to be a subdirectory of `/mnt/tdm/rich/expansions/`.

### By hand

Theses are located in `/mnt/tdm/rich/expansions/`.

To get all the aero_astro theses, plus their associated xml metadata, in a reasonably efficient way:

```
ssh m31@repo-dev-1.mit.edu "find /mnt/tdm/rich/expansions/aero_astro/ -name '*-new.txt' -or -name '*.xml' > tempfile.txt ; tar -czvf tempfile.tar.gz -T tempfile.txt" ; scp m31@repo-dev-1.mit.edu:tempfile.tar.gz files/
```

(This will prompt for your password twice, and you'll probably want to ssh over to delete tempfile.* at some point.)

Modify the path fed into find if you want a different department (or all the departments, but expect that to take some time and require a lot of space.)

When you untar it, you'll want to do
`tar -xzf tempfile.tar.gz -s '|.*/||'`

The regex at the end removes the directory structure from the filenames (it serves no useful purpose for us and only makes life harder, so let's kill it).
