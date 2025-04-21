![MIT License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8--3.12-blue)

# ChatPDF: Local AI-Powered Research Assistant for PDFs

A fast, offline-friendly CLI tool for indexing and querying large collections of research papers using OpenAI's models, FAISS, and (optionally) Grobid.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Index a folder of PDFs
chatpdf index ./papers myindex

# Ask a question!
chatpdf query myindex "What is the main result in the KPZ paper?"

## Setup

- Create and activate a Python virtual environment (optional but recommended)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set your OpenAI API key:
  - Via environment variable:
    ```bash
    export OPENAI_API_KEY="your_api_key_here"
    ```
  - Or use the built-in auth command:
    ```bash
    chatpdf auth login
    ```
    This will prompt you for your key and store it in `~/.config/chatpdf/.env`.

### Grobid Setup (optional)

To enable structured section extraction with Grobid, run a Grobid server locally:

1. Ensure you have Java 11+ and Docker installed.
2. Pull and run the official Docker image:
   ```bash
   docker pull lfoppiano/grobid:0.7.3
   docker run --rm -t -p 8070:8070 lfoppiano/grobid:0.7.3
   ```
3. Optionally, set the `GROBID_URL` environment variable to your server URL:
   ```bash
   export GROBID_URL=http://localhost:8070
   ```

When running the `index` command, pass `--grobid` to use this service for sectioning and metadata.

## Installation

Install via pip:
```bash
pip install .
# or in editable mode for development:
pip install -e .
```

## Configuration

You can customize model choices and chunking parameters via a JSON configuration file.
ChatPDF looks for configuration in two locations:

- Local: `./.chatpdf.json`
- Global: `~/.config/chatpdf/config.json` (created by `chatpdf auth login`)

Supported configuration keys:
- `embedding_model`: OpenAI embedding model name (default: "text-embedding-ada-002").
- `chat_model`: OpenAI chat model name for Q&A (default: "gpt-3.5-turbo").
- `chunk_size`: Number of tokens per chunk (default: 500).
- `chunk_overlap`: Token overlap between chunks (default: 50).

Example `~/.config/chatpdf/config.json`:
```json
{
  "embedding_model": "text-embedding-ada-002",
  "chat_model": "gpt-4",
  "chunk_size": 800,
  "chunk_overlap": 100
}
```

### Recommended OpenAI Models
When choosing models, you should balance cost, speed, and quality. Here are some common options:

Embedding models:
- text-embedding-ada-002 (default): very cost-effective and high-quality for most tasks
- text-embedding-3-small: lower cost and faster, suitable for large batch embeddings
- text-embedding-3-large: higher accuracy for specialized applications

Chat / reasoning models:
- gpt-3.5-turbo (default): fast and affordable general-purpose chat
- gpt-3.5-turbo-16k: extended context window (up to 16k tokens)
- gpt-4: top-tier reasoning quality, standard-window (~8k tokens)
- gpt-4-32k: extended context window (up to 32k tokens) for very long conversations

Specify your choice in the config file under `embedding_model` and `chat_model`.

## Tunable Parameters
In addition to model choice, you can adjust these parameters:

- `chunk_size` (config file): number of tokens per text chunk (default: 500)
- `chunk_overlap` (config file): token overlap between chunks (default: 50)
- `top_k` (CLI flag `--top_k`): number of top document chunks to retrieve (default: 5)
- `temperature` (CLI flag `--temperature`): sampling temperature for the chat model (default: 0.2).
  • Lower values (0.0–0.3) produce more deterministic, focused responses—ideal for factual Q&A.
  • Medium values (0.4–0.7) balance creativity and accuracy.
  • Higher values (0.8–1.0) yield more varied or imaginative outputs, but may introduce hallucinations.
  Typical practice for research–centric answers is 0.2–0.5.

Examples:
```bash
# Use top 10 chunks and a higher sampling temperature
chatpdf query myindex "Explain the main theorem" --top_k 10 --temperature 0.7

# In REPL mode, adjust temperature and top_k
chatpdf repl myindex --top_k 8 --temperature 0.3
```

## Usage

1. Index your PDFs (creates a FAISS index and metadata):
   ```bash
   chatpdf index <root_directory> <index_prefix> [--grobid] [--grobid-url URL] [--semantic]
   ```
   Options:
   - `--grobid` to use Grobid for structured section extraction (requires Grobid service running)
   - `--grobid-url URL` to specify the Grobid server URL (default: http://localhost:8070)
   - `--semantic` to enable Unstructured-based paragraph tokenization and semantic chunking
   This will walk through `<root_directory>`, extract and embed text chunks, and save:
   - `<index_prefix>.pkl` (documents metadata)
   - `<index_prefix>.faiss` (FAISS vector index)

2. Query your PDF collection (uses FAISS for fast retrieval):
   ```bash
   chatpdf query <index_prefix> "Your question here"
   ```
   This will print the answer (with any mathematical formulas formatted in LaTeX notation) and a list of references (file path and page number) for each retrieved chunk.

3. Interactive REPL mode:
   ```bash
   chatpdf repl <index_prefix>
   ```
   Starts a session where you can ask multiple follow-up questions against the same index. Type `exit` or press Ctrl-D to quit.


## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

> Note: Use of OpenAI models must comply with [OpenAI's API Terms of Use](https://openai.com/policies/terms-of-use).
