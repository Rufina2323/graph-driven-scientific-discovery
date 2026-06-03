# Graph-Driven Scientific Discovery

This repository contains the code for the thesis project **“Enhancing Scientific Idea Generation and Paper Writing with Knowledge Graph Link Prediction and Large Language Models.”**

## Goal

The goal of this project is to support **scientific idea generation** by predicting promising future connections between research concepts and using them to generate draft research proposals.

## Problem

Scientific literature is growing rapidly, making it difficult for researchers to manually identify new and valuable research directions. Existing LLM-based systems can generate text, but they still depend heavily on the initial idea or topic provided to them.

## Approach

The project builds a pipeline that:

1. extracts scientific concepts from papers;
2. builds a temporal concept co-occurrence graph;
3. predicts future links between concepts;
4. uses top-ranked predicted links to generate research proposal drafts with LLMs.

## Dataset

The experiments use **11,349 accepted ICLR papers from 2017–2025**.

## Main Results

- Best concept extraction method: **LLM-based extraction**
- Best link prediction model: **GAT + MLP**
- Generated drafts achieved a mean weighted score of **5.84 / 10**

## Use Cases

This pipeline can be used for:

- research ideation support;
- literature exploration;
- proposal brainstorming;
- discovering emerging research directions.

---

## Project Structure
```bash
graph-driven-scientific-discovery/
│
├── configs/ # YAML configuration files for every pipeline stage
│ ├── concept_extracting.yaml # → concept extraction settings
│ ├── concept_normalization.yaml # → concept normalization settings
│ ├── dataset_creation.yaml # → dataset / knowledge-graph construction settings
│ ├── model_training.yaml # → link-prediction model settings
│ ├── draft_generation.yaml # → draft generation settings (pairs CSV, model, etc.)
│ └── draft_assessment.yaml # → draft assessment settings
│
├── data/ # All data artefacts (raw, interim, and processed)
│ └── README.md # Instruction how to download data
│
├── notebooks/ # Jupyter notebooks for exploration and analysis
│ ├── images/ # → figures and plots used in / produced by notebooks
│ └── *.ipynb # → analysis notebooks
│
├── src/ # Source code (Python package)
│ ├── init.py
│ ├── extract_concepts.py # → Stage 1: extract concepts from papers
│ ├── normalize_concepts.py # → Stage 2: normalise / deduplicate concepts
│ ├── create_concept_datasets.py # → Stage 3: build knowledge-graph datasets
│ ├── dataset_construction/ # → DNE-based dataset helpers (see Setup below)
│ │ ├── README.md
│ │ └── DNE/ # → cloned from https://github.com/rui-yan/DNE
│ ├── run_link_prediction.py # → Stage 4: train & evaluate link-prediction models
│ ├── generate_draft.py # → Stage 5: LLM-powered draft generation
│ ├── assess_draft_with_llm.py # → Draft evaluation with an LLM judge
│ └── utils/ # → shared helpers (config loading, API wrappers, etc.)
│
├── pyproject.toml # Project metadata & dependencies
├── uv.lock # Locked dependency snapshot (uv)
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python = 3.14** | Tested with 3.14 |
| **[uv](https://github.com/astral-sh/uv)** | Fast Python package manager used to run every stage |
| **API keys** | `ANTHROPIC_API_KEY` for Claude and / or `OPENAI_API_KEY` for GPT (needed by stages 5 & 6) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rufina2323/graph-driven-scientific-discovery.git
cd graph-driven-scientific-discovery
```

### 2. Install dependencies
```bash
uv sync
```

This reads pyproject.toml + uv.lock and creates a virtual environment with all pinned dependencies.

### 3. Clone the DNE repository (required for dataset construction)
```bash
cd src/dataset_construction
git clone https://github.com/rui-yan/DNE.git
cd ../..
```

### 4. Set API keys (for draft generation & assessment)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # if using Claude
export OPENAI_API_KEY="sk-..."          # if using GPT
```

## Configuration
Every stage reads a YAML file from configs/. The default path is built in, so you only need --config when you want to override it:

```bash
uv run python -m src.extract_concepts --config configs/extract_concepts.yaml
```
Edit the YAML files to change data paths, model parameters, number of examples, etc.

## Pipeline Stages
Run stages sequentially. Each stage reads outputs of the previous one.

**Stage 1 — Extract Concepts**
```bash
uv run python -m src.extract_concepts
```
Extracts raw concepts from scientific papers.

**Stage 2 — Normalize Concepts**
```bash
uv run python -m src.normalize_concepts
```
Deduplicates and normalizes extracted concepts.

**Stage 3 — Create Concept Datasets**
```bash
uv run python -m src.create_concept_datasets
```
Builds knowledge-graph-ready datasets from normalized concepts.

**Stage 4 — Run Link Prediction**
```bash
uv run python -m src.run_link_prediction
```
Trains and evaluates link-prediction models on the concept graph to surface novel concept pairs.

**Stage 5 — Generate Drafts**
```bash
uv run python -m src.generate_draft
```

**Stage 6 — Assess Drafts**
```bash
uv run python -m src.assess_draft_with_llm
```