# Streamline Seeding Retrieval

Local retrieval for structured streamline seeding application records.

The tooling covers:

- loading application records from `knowledge_base/records/tagged_applications.jsonl`
- auditing schema, IDs, tags, vocabulary coverage, and text-quality issues
- loading `knowledge_base/vocabulary/tag_vocabulary.json` or deriving the observed vocabulary from the data
- deterministic query tagging from the controlled vocabulary
- query tagging reuses the same local normalization rules that tagged the corpus records
- local index building with either BGE sentence-transformer embeddings or a deterministic hashing/TF-IDF fallback
- hybrid retrieval with embedding similarity, tag overlap, compatibility penalties, and score explanations

## Data Audit

```powershell
.\.venv\Scripts\python.exe -m rag.cli.audit_data
```

The default report path is:

```text
artifacts/reports/data_audit.json
```

## Vocabulary Derivation

Print the vocabulary observed in the current corpus:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_vocabulary
```

Write the derived vocabulary to a file:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_vocabulary --output artifacts/reports/derived_tag_vocabulary.json
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
.\.venv\Scripts\python.exe -m rag.cli.build_index --embedding-backend sentence-transformers --model-name BAAI/bge-small-en-v1.5
```

After the model has been downloaded once, rebuild fully offline from the local cache:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_index --embedding-backend sentence-transformers --model-name BAAI/bge-small-en-v1.5 --local-files-only
```

For development only, a deterministic hashing/TF-IDF fallback is still available:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_index --embedding-backend local-hashing
```

The index files are:

```text
artifacts/indexes/default/records.jsonl
artifacts/indexes/default/embeddings.npy
artifacts/indexes/default/metadata.json
```

## Retrieval

```powershell
.\.venv\Scripts\python.exe -m rag.cli.retrieve "I have a 3D CFD flow field and want to seed streamlines around vortices without too much clutter." --pretty
```

Full JSON is printed by default:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.retrieve "I need streamline seeds for 2D critical points"
```

Input-query tags are produced by applying the same corpus normalization rule set to the query text. This keeps query tags and record tags aligned; no separate query-only regex vocabulary is used.

## Local VTK Seeding RAG App

Launch the desktop application:

```powershell
.\.venv\Scripts\python.exe -m app.cli.launch
```

The app provides:

- a structured visualization request form aligned with the retrieval records
- VTK-native dataset metadata extraction for `.vtk`, `.vti`, `.vtu`, `.vtp`, `.vts`, and `.vtr`
- local retrieval over the existing seeding index
- optional RAG use, allowing final prompts to be built either with retrieved seeding records or only from dataset metadata and user intent
- final prompt assembly from dataset metadata, user intent, and selected structured record fields when RAG is enabled
- LiteLLM-based provider/model switching for code generation, including OpenAI, Anthropic, Gemini, Blablador, and custom LiteLLM-compatible endpoints
- default code generation with `anthropic/claude-opus-5`, a 50,000 token output budget adjustable up to 200,000 tokens, automatic continuation on truncation, and two validation-repair attempts
- compact terminal logging of each LLM call stack, including finish reasons, provider-reported input/output token counts, and aggregate token totals when available
- explicit confirmation before generated VTK code is executed
- isolated generated-code smoke testing in a subprocess so VTK/Qt crashes do not terminate the main app
- a safe PNG preview rendered by the subprocess
- an interactive VTK viewport launched in a monitored child process; if that window crashes, the main app stays open and reports the child-process exit
- interactive viewport stdout/stderr mirrored to the launching terminal for easier copying/debugging
- automatic last-session restore for the dataset path, request fields, retrieval state, prompt, generated code, and non-secret LLM settings
- manual session save/load from the workflow panel, with saved filenames containing the selected dataset name, model, query mode, RAG mode, and save timestamp
- experiment metadata fields for comparing configurations, including a success/failure result flag, with manual session saves copying the current viewport preview to a matching `_viewport.png` file when available

The generated code must define:

```python
def create_visualization(dataset_path: str, metadata: dict, user_request: dict):
    """Return a vtkRenderer containing the complete visualization."""
```

Generated prompts, retrieval snapshots, LLM responses, and code are stored under ignored `local/generated/` folders for reproducibility.
Autosaved and manually saved GUI sessions are stored locally under ignored `local/sessions/`. API keys are not saved.

In the LLM panel, choose a provider and enter the matching API key, or leave the key field empty when the provider-specific environment variable is already set. The app only checks the API key for the currently selected/resolved model provider. The model field remains editable, so any LiteLLM model string supported by your installed LiteLLM version can be used.

For Blablador, choose the `Blablador` provider. The default model is `alias-code`, and the app uses Blablador's OpenAI-compatible endpoint at `https://api.blablador.fz-juelich.de/v1/` unless you enter a custom API base. To provide the token through PowerShell for the current terminal session:

```powershell
$env:BLABLADOR_API_KEY = "your-blablador-token"
```

To persist it for future PowerShell sessions:

```powershell
[Environment]::SetEnvironmentVariable("BLABLADOR_API_KEY", "your-blablador-token", "User")
```

The current-session form only works for apps launched from that same PowerShell process after setting the variable. On Windows, the app also checks persistent User/Machine environment variables directly, so the persistent form is usually less fragile. The canonical spelling is `BLABLADOR_API_KEY`; `BLABLADOOR_API_KEY` is also accepted as a compatibility alias.

The token is the Helmholtz Codebase personal access token described in the Blablador API guide: https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/

## Evaluation

Run the curated example queries:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.evaluate
```

The default report path is:

```text
artifacts/reports/evaluation.json
```

## Tests

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

VTK-dependent tests are skipped when `vtk` is not installed.

## Offline Note

Retrieval does not call an LLM. The production path uses local `sentence-transformers` with `BAAI/bge-small-en-v1.5`, stores normalized embeddings in `artifacts/indexes/default/embeddings.npy`, and uses exact dot-product search over the local matrix. For 122 records, exact matrix search is faster, simpler, and more accurate than an approximate ANN/vector database.

Query-time model loading is forced to `local_files_only=True`, so once the model is cached, retrieval stays local/offline.
