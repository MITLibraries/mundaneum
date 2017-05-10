# 1_tokenized.model

## The plan:
* make the script run at the command line, with an option for specifying model name
* add nontrivial tokenization

## Results
Script version as of: `b244f6d`
Trained on: a subset of aero_astro theses
Runtime: about 10 minutes
No tokenization or preprocessing beyond split()

## Notes
OK. I lost a lot of this due to a git surgery problem, but the overall was:

* The reference aero_astro thesis had overall similar results for a most_similar query (same top few, somewhat different below, different levels of certainty in a way I'd have to visualize to understand)
* You can load saved models with gensim.models.Doc2Vec.load(filename)
* models.wv.most_similar gives word vector similarity
* model.wv.most_similar(positive=['apollo', 'russian'], negative=['american']) yields 'vostok' (this is "apollo minus american plus russian")
*

# tdm.model

## Results
Script version as of: `b244f6d`
Trained on: a subset of aero_astro theses
Runtime: about 10 minutes
No tokenization or preprocessing beyond split()

## Notes
The results are actually interesting!
`friction`, `friction.`, and `Friction` are classed as similar
So are `military` and `commercial`
Most-similar theses at ~25% similarity are not similar
But most-similar theses at ~65% *are* highly similar; e.g. there's a cluster of aviation-regulatory-alliance theses that show up as very like one another
