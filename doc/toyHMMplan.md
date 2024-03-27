# Preprocessing
## Binning
1. _First_ run through genome _base by base_, learn segment transition probabilities.
2. Save _max likelihood_ set of segments?

#### Parsing `bedGraph`s
- ~30-40 chromosomes, in every signal track/assay, for each experiment
- Need to...
	1. Construct `pyBedGraph.BedGraph(chrom_sizes, bedgraph_file, [chrom_names])`
	2. Load each chromosome for each `BedGraph`
	3. Obtain means for intervals over every chromosome
 
#### Parsing `bigWig`s
Using [pyBigWig](https://github.com/deeptools/pyBigWig)
```python
>>> bw.stats("1",99, 200, type="max", nBins=2)
[1.399999976158142, 1.5]
```
--> Can just use entire chromosome, $$\mathtt{nBins} := \lceil \frac{|chrom|}{|bin|} \rceil.$$

>[!todo]

- [ ] transpose binned matrix --> positions x tracks
- [ ] check output of pyBigWig
	- [ ] if resolution of different segments different, simple arithmetic mean inaccurate
- [ ] ensure final chromosomes' bin "series" contain the data of exactly its coordinates interval

## Model Creation
In general, copy Segway.
- Constraint: `(num binned signals)` = `(num segments)`

#### Emission Model
A distribution for each chromatin state _within every track_.

To retrieve, throw all emission nodes' params into $k \times d$ Tensors:
- Means: `d_ij.means`,
- Variances: `d_ij.covs`
#### Transition Model
Discrete categorical--categories are each possible chromatin state.
|> Initialize to uniform?

---

_Mar 22, 2024_
1. Add option directly provide fraction-of-genome tensors _already minibatched_
2. --> ENCODE Pilot: run SimpleAGA on ^