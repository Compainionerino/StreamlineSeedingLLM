# Streamline Seeding Retrieval

Local retrieval for structured streamline seeding application records.

The tooling covers:

- loading application records from `tagged_applications.jsonl`
- auditing schema, IDs, tags, vocabulary coverage, and text-quality issues
- loading `tag_vocabulary.json` or deriving the observed vocabulary from the data
- deterministic query tagging from the controlled vocabulary
- query tagging reuses the same local normalization rules that tagged the corpus records
- local index building with either BGE sentence-transformer embeddings or a deterministic hashing/TF-IDF fallback
- hybrid retrieval with embedding similarity, tag overlap, compatibility penalties, and score explanations

## Data Audit

```powershell
python audit_data.py --data tagged_applications.jsonl --vocabulary tag_vocabulary.json
```

The default report path is:

```text
reports/data_audit.json
```

## Vocabulary Derivation

Print the vocabulary observed in the current corpus:

```powershell
python build_vocabulary.py --data tagged_applications.jsonl
```

Write the derived vocabulary to a file:

```powershell
python build_vocabulary.py --data tagged_applications.jsonl --output reports/derived_tag_vocabulary.json
```

## Index Building

Create and populate the project virtual environment:

```powershell
C:\Users\Fabian\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Build the production neural index with BGE:

```powershell
.\.venv\Scripts\python.exe build_index.py --data tagged_applications.jsonl --vocabulary tag_vocabulary.json --embedding-backend sentence-transformers --model-name BAAI/bge-small-en-v1.5
```

After the model has been downloaded once, rebuild fully offline from the local cache:

```powershell
.\.venv\Scripts\python.exe build_index.py --embedding-backend sentence-transformers --model-name BAAI/bge-small-en-v1.5 --local-files-only
```

For development only, a deterministic hashing/TF-IDF fallback is still available:

```powershell
.\.venv\Scripts\python.exe build_index.py --embedding-backend local-hashing
```

The index files are:

```text
index/records.jsonl
index/embeddings.npy
index/metadata.json
```

## Retrieval

```powershell
.\.venv\Scripts\python.exe retrieve.py "I have a 3D CFD flow field and want to seed streamlines around vortices without too much clutter." --pretty
```

Full JSON is printed by default:

```powershell
.\.venv\Scripts\python.exe retrieve.py "I need streamline seeds for 2D critical points"
```

Input-query tags are produced by applying the same corpus normalization rule set to the query text. This keeps query tags and record tags aligned; no separate query-only regex vocabulary is used.

## Evaluation

Run the curated example queries:

```powershell
.\.venv\Scripts\python.exe evaluate.py --index-dir index --queries eval_queries.json
```

The default report path is:

```text
reports/evaluation.json
```

## Offline Note

Retrieval does not call an LLM. The production path uses local `sentence-transformers` with `BAAI/bge-small-en-v1.5`, stores normalized embeddings in `index/embeddings.npy`, and uses exact dot-product search over the local matrix. For 122 records, exact matrix search is faster, simpler, and more accurate than an approximate ANN/vector database.

Query-time model loading is forced to `local_files_only=True`, so once the model is cached, retrieval stays local/offline.
