# ChatPDF: Local Research Assistant for PDF Documents

A CLI tool for indexing and querying your local PDF collection using OpenAI APIs.

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
   This will print the answer and a list of references (file path and page number) for each retrieved chunk.

3. Interactive REPL mode:
   ```bash
   chatpdf repl <index_prefix>
   ```
   Starts a session where you can ask multiple follow-up questions against the same index. Type `exit` or press Ctrl-D to quit.
