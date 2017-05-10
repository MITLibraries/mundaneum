# 1_tokenized.model

The plan:
* make the script run at the command line, with an option for specifying model name
* add nontrivial tokenization

# tdm.model

Script version as of: `b244f6d`
Trained on: a subset of aero_astro theses
No tokenization or preprocessing beyond split()

Results:
Actually interesting!
`friction`, `friction.`, and `Friction` are classed as similar
So are `military` and `commercial`
Most-similar theses at ~25% similarity are not similar
But most-similar theses at ~65% *are* highly similar; e.g. there's a cluster of aviation-regulatory-alliance theses that show up as very like one another
