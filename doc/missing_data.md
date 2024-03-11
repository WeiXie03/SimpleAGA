Some regions in signal track files have no data. Often underlying this is reads that cannot be uniquely mapped to a region.

$$P(Q_t|X_t) = \frac {P(X_t|Q_t)P(Q_t)} {P(X_t)}$$

# Segway's Approach
### Indicators


"Missing" signal value at position $t$ means observation $X_t = x_t$ not available.
Ok, after "sever conditional dependence", what substitute for $P(X_t|Q_t)$?
"Marginilization" $\implies$ just $P(Q)$, depends _only on ___transitions___ $P(Q_t|Q_{t-1})$, and next $P(Q_{t+1}|Q_t)$?

Yeah, just:
- only $\set{t}$ with _available_ $x_t$'s in Expectation Maximization, and
- affect transitions learning through $>1$-length event "chains" that end in emitting nodes

- [ ] Track _locations_ of NaN's, ignore these when inferring from observations.
	- [ ] When deciding which _bins_ to set to NaN, maybe just use a threshold on proportion of missing within bin
_June 19, 2023_
^ turns out cannot simply feed pomegranate HMM NaN's, None's for missing observations
=> just omit, literally shorten track fed in by cutting out?

### Track Weighting
For each $i$th track, define $N^{(i)} := \sum_t \mathring{X_t}$, where $\mathring{X_t} = 1$ if there is an observation for $t$ and 0 otherwise. Then _normalize_ each track's contribution to the hidden state by "proprotionalizing" to the non-missing data content/ratio in each: $$P(X_t|Q_t) := P(X_t|Q_t) ^ {\frac {N^{(i)}} {N^*}}, ~ t=1,...,T$$ where $N^* = \max_j (N^{(j)})$ and $T$ is the total number of bins.