# Preprocessing
## Segmenting
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

## Model Creation
- Constraint: `(num binned signals)` = `(num segments)`
- [ ] check transition model type (i.e. Bernoullli?)