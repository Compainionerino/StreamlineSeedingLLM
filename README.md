# Streamline Seeding LLM/RAG Prototype

This project explores whether retrieval-augmented large language models can generate better VTK streamline visualizations. It combines a local retrieval system for streamline seeding knowledge with a desktop application that prompts an LLM, validates the generated Python code, runs it safely, and stores experiment sessions for later analysis.

## Feature Overview

### Desktop Application

The desktop application is the main workflow for experiments. It loads VTK datasets, assembles prompts with or without retrieved seeding knowledge, generates visualization code through LiteLLM, validates and executes that code in a subprocess, and saves experiment sessions for later comparison.

### RAG System

The RAG system indexes structured application records from `knowledge_base/records/tagged_applications.jsonl`. Each record describes a streamline seeding or placement method, including the data type, target visualization task, seeding strategy, parameters, limitations, and source evidence.

At query time, the system:

- tags the user query with the same controlled vocabulary used by the corpus
- retrieves relevant seeding records from a local embedding index
- combines embedding similarity, tag overlap, compatibility checks, and penalties
- returns matching records with score explanations

Retrieval is local. It does not call an LLM. The default index can use `sentence-transformers` with `BAAI/bge-small-en-v1.5`; a deterministic local hashing backend is also available for development.

### Experiment Dashboard

The experiment dashboard aggregates saved sessions into a local website. It shows raw runs, primary runs, RAG-vs-no-RAG pairs, explorative-vs-feature-aware pairs, grouped summaries, duplicate conditions, missing conditions, and feature-rubric seed data.

This is useful for inspecting the thesis results and correcting evaluation metadata without manually editing every JSON file.

## Setup

Create and populate the project virtual environment:

```powershell
C:\Users\Fabian\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running the Application

Launch the desktop application:

```powershell
.\.venv\Scripts\python.exe -m app.cli.launch
```

In the app:

1. Choose a dataset.
2. Fill in the visualization request.
3. Enable or disable RAG.
4. Select the LLM provider and model.
5. Generate code.
6. Review, execute, and save the experiment session.

The app extracts dataset metadata, assembles the final prompt, calls the selected LiteLLM provider, validates the generated code, and executes it in a protected subprocess. Successful runs can be saved with a viewport preview and experiment metadata.

The generated code must define:

```python
def create_visualization(dataset_path: str, metadata: dict, user_request: dict):
    """Return a vtkRenderer containing the complete visualization."""
```

Generated prompts, retrieval snapshots, LLM responses, and code are stored under `local/generated/`. Autosaved and manually saved GUI sessions are stored under `local/sessions/`. API keys are not saved.

## Using the RAG System from the Command Line

Build the default retrieval index:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_index
```

Build the BGE index explicitly:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_index --embedding-backend sentence-transformers --model-name BAAI/bge-small-en-v1.5
```

After the model has been downloaded once, rebuild from the local cache:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_index --embedding-backend sentence-transformers --model-name BAAI/bge-small-en-v1.5 --local-files-only
```

The index files are:

```text
artifacts/indexes/default/records.jsonl
artifacts/indexes/default/embeddings.npy
artifacts/indexes/default/metadata.json
```

Run retrieval for a query:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.retrieve "I have a 3D CFD flow field and want to seed streamlines around vortices without too much clutter." --pretty
```

Without `--pretty`, the command prints the full JSON payload:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.retrieve "I need streamline seeds for 2D critical points"
```

For development only, a deterministic hashing/TF-IDF fallback is available:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.build_index --embedding-backend local-hashing
```

## API Keys

API keys can be entered in the app, or provided through environment variables. The app only checks the key for the currently selected provider.

Supported provider variables:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GEMINI_API_KEY
GOOGLE_API_KEY
BLABLADOR_API_KEY
BLABLADOOR_API_KEY
```

For the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
$env:ANTHROPIC_API_KEY = "your-anthropic-key"
$env:GEMINI_API_KEY = "your-gemini-key"
$env:BLABLADOR_API_KEY = "your-blablador-token"
```

To persist it for future PowerShell sessions:

```powershell
[Environment]::SetEnvironmentVariable("BLABLADOR_API_KEY", "your-blablador-token", "User")
```

Blablador uses the OpenAI-compatible endpoint `https://api.blablador.fz-juelich.de/v1/` by default. The canonical variable is `BLABLADOR_API_KEY`; `BLABLADOOR_API_KEY` is accepted as a compatibility alias.

## Experiment Dashboard and Editor

Aggregate saved experiment sessions and build a local comparison dashboard:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.analyze_experiments
```

The dashboard and analysis exports are written to:

```text
artifacts/reports/experiment_dashboard/index.html
artifacts/reports/experiment_dashboard/data/
```

The generated data folder includes raw runs, one primary run per condition, RAG-vs-no-RAG pairs, explorative-vs-feature-aware pairs, grouped summaries, and a dataset feature rubric template. Duplicate conditions are resolved with the latest saved session by default; pass `--primary-strategy best` to select the highest-scoring run per condition instead.

To edit documented experiment fields directly from the dashboard, launch the local editor server:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.experiment_editor
```

Then open the printed localhost URL. Edit buttons appear in the dashboard only in this editor mode. Saving updates the selected session JSON under `local/sessions/`, creates a timestamped backup under `local/sessions/_backups/`, regenerates the dashboard data, and refreshes the page. The editor is limited to the experiment evaluation fields: result, attempts, recognized features, feature notes, colormap use, suggested seeding use, seeding score, and seeding notes.

## Useful Development Commands

Run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

VTK-dependent tests are skipped when `vtk` is not installed.

Run the data audit:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.audit_data
```

Run the curated retrieval evaluation:

```powershell
.\.venv\Scripts\python.exe -m rag.cli.evaluate
```
