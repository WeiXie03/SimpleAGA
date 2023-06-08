Some regions in signal track files have no data. Often underlying this is reads that cannot be uniquely mapped to a region.

$$P(Q_t|X_t) = \frac {P(X_t|Q_t)P(Q_t)} {P(X_t)}$$

# Segway's Approach
### Indicators

Ok, after "sever conditional dependence", what substitute for $P(X_t|Q_t)$?
"Marginilization" $\implies$ just $P(Q)$, depends _only on ___transitions___ $P(Q_t|Q_{t-1})$, and next $P(Q_{t+1}|Q_t)$?

### Track Weighting
For each $i$th track, define $N^(i) := \sum_t \mathring{X_t}$. Then _normalize_ each track's contribution to the hidden state by "proprotionalizing" to the non-missing data content/ratio in each: $$P(X_t|Q_t) := P(X_t|Q_t) ^ {\frac {N^(i)} {N^*}}, ~ t=1,...,T$$ where $N^* = \max_j(N^(j))$ and $T$ is the total number of bins.