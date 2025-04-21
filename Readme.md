# ChatPDF: Local Research Assistant for PDF Documents

A CLI tool for indexing and querying your local PDF collection using OpenAI APIs.

## Setup

- Create and activate a Python virtual environment (optional but recommended)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Set your OpenAI API key:
  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  ```

## Usage

1. Index your PDFs (creates a FAISS index and metadata):
   ```bash
   python3 chatpdf.py index <root_directory> <index_prefix>
   ```
   This will walk through `<root_directory>`, extract and embed text chunks, and save:
   - `<index_prefix>.pkl` (documents metadata)
   - `<index_prefix>.faiss` (FAISS vector index)

2. Query your PDF collection (uses FAISS for fast retrieval):
   ```bash
   python3 chatpdf.py query <index_prefix> "Your question here"
   ```
   This will print the answer and a list of references (file path and page number) for each retrieved chunk.

3. Interactive REPL mode:
   ```bash
   python3 chatpdf.py repl <index_prefix>
   ```
   Starts a session where you can ask multiple follow-up questions against the same index. Type `exit` or press Ctrl-D to quit.
