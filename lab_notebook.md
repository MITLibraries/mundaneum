# future work ideas
Can you plot your min cost function after each training epoch, to get a sense of whether you have a reasonable number of epochs/good alpha value?

Can you make disciplinary filters by looking at the strength of correlations of target words in different fields? Like in engineering, hydrogen is a lot like oxygen, but in biology, it's not, and in chemistry, it's somewhere in between - this field's perspective is closer to this one than that?

# 1_tokenized_{dlc}.model

## The plan
* train on different departments that might care about 'oxygen'
* see what their different views of its meaning are
* along the way: upgrade script to fetch files over the network

This was inspired by 1_tokenized.model, wherein 'oxygen' is near 'hydrogen', 'nitrogen', 'propellant', and 'hypergolic' -- that is, a concept cluster roughly meaning 'rocket fuel' -- and not 'elements', as H and N might indicate. However, I hypothesize that, in the chemistry department, 'oxygen' is part of a cluster that means something closer to 'element' (and, in particular, in both biology and chemistry it is likely to be close to 'carbon', which is not the case in aero_astro).

It turns out, in fact, that in chemistry, the near neighbors of 'oxygen' are elements - ! And in biology, the cluster looks like "energy" (fuel for the body; words like 'energy', 'nutrient', and 'atp' are nearby).

In physics, it isn't at all obvious.

## Results

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
