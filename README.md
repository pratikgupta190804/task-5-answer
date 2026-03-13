# Task 5 — Is It Truly Deterministic?

## Experiment Setup

To verify the claim that the tokenizer training process is deterministic, the same tokenizer was trained **twice** using:

* **Model:** BPE
* **Vocabulary Size:** 200
* **Corpus:** `data/corpus.txt`
* **Configuration:** identical for both runs

Two independent training runs were executed:

```
artifacts/run1
artifacts/run2
```

Both models were then used to encode the same test input.

---

# Experiment 

Script to compare encoded output of both models
```bash
@'
from pathlib import Path
from abctokz import Tokenizer
import hashlib

run1 = r"artifacts\run1"
run2 = r"artifacts\run2"
test_file = r"data\test.txt"

tok1 = Tokenizer.load(run1)
tok2 = Tokenizer.load(run2)

text = Path(test_file).read_text(encoding="utf-8")

enc1 = tok1.encode(text)
enc2 = tok2.encode(text)

print("Encoded tokens identical:", enc1 == enc2)

print("\nRun1 tokens:", enc1)
print("\nRun2 tokens:", enc2)

def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

files = ["vocab.json", "merges.txt"]

print("\nFile Comparisons")
for f in files:
    p1 = Path(run1) / f
    p2 = Path(run2) / f

    if p1.exists() and p2.exists():
        h1 = file_hash(p1)
        h2 = file_hash(p2)
        print(f"{f}: identical =", h1 == h2)
    else:
        print(f"{f}: file missing")
'@ | python -
```

Output

```
Encoded tokens identical: True

Run1 tokens: Encoding(n_tokens=99, tokens=['J','##a','##n','##a','## ', ...])

Run2 tokens: Encoding(n_tokens=99, tokens=['J','##a','##n','##a','## ', ...])

File Comparisons
vocab.json: identical = True
merges.txt: identical = True
```

Both runs also produced the same artifact files:

```
config.json
manifest.json
merges.txt
special_tokens.json
vocab.json
```

---

# What Parts Are Deterministic?

The experiment confirms that several components of the tokenizer pipeline are **fully deterministic**.

### 1. Vocabulary Generation

The `vocab.json` files were identical across runs.

This means that:

* Token frequencies were computed consistently
* Subword merges were learned in the same order
* Token IDs were assigned deterministically

---

### 2. Merge Rule Learning (BPE)

The `merges.txt` files were identical.

This indicates that the **BPE merge algorithm consistently selected the same most frequent pair at every step**, leading to the same merge sequence.

---

### 3. Encoding Output

Encoding the same input text produced:

* identical token sequences
* identical token IDs
* identical token counts

This confirms that **inference is deterministic when the model artifacts are identical**.

---

# What Parts Are Not Strictly Deterministic?

Even though the tokenizer itself behaves deterministically, some aspects of the process may vary slightly.

### Benchmark Timing

Training time and encoding speed may differ slightly between runs due to:

* CPU scheduling
* background processes
* system load

This does **not affect tokenizer correctness**, so it is acceptable.

---

# Remaining Risks — When Could Results Differ?

Even deterministic algorithms can produce different outputs under certain conditions.

### 1. Corpus Order Changes

If the corpus lines are shuffled, the frequency counts might be processed in a different order during tie situations, which could alter merge decisions.

---

### 2. Frequency Tie-Breaking

If two character pairs have **exactly the same frequency**, the algorithm must choose one first.

If the implementation does not enforce a deterministic tie-breaking rule, merge order could change.

---

### 3. Different Software Versions

Using different versions of:

* the tokenizer library
* Python
* dependencies

could potentially affect behavior.

---

### 4. Parallel Processing

If training were parallelized in the future, race conditions could introduce non-deterministic ordering unless explicitly controlled.

---

# Conclusion

The experiment demonstrates that the tokenizer training pipeline is **deterministic under controlled conditions**.

Training the same tokenizer twice with the same corpus and configuration produced:

* identical vocabularies
* identical merge rules
* identical encoded outputs

The only non-deterministic aspects are **external factors such as runtime performance**, which do not affect the correctness of the tokenizer.
