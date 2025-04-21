#!/usr/bin/env python3
"""
Command-line tool to chat with a collection of local PDF documents using OpenAI APIs.
"""
__version__ = '0.1.0'
"""
Command-line tool to chat with a collection of local PDF documents using OpenAI APIs.
"""
import os
import sys
import argparse
import pickle

from dotenv import load_dotenv
import numpy as np
import faiss
import tiktoken

# OpenAI model constants
EMBED_MODEL = 'text-embedding-ada-002'
CHAT_MODEL = 'gpt-3.5-turbo'
# Token-based chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
import openai
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    openai.api_key = api_key



def extract_text_from_pdf(path):
    """Extract text from a PDF file, skipping broken files."""
    text_chunks = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
        return "\n".join(text_chunks)
    except PdfReadError as e:
        print(f"⚠️ Skipping unreadable PDF: {path} ({e})")
        return ""
    except Exception as e:
        print(f"⚠️ Unexpected error reading {path}: {e}")
        return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into token-based chunks with overlap."""
    # Initialize tokenizer for chat model
    try:
        encoder = tiktoken.encoding_for_model(CHAT_MODEL)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")
    # Encode to tokens
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    total = len(tokens)
    # Slide window over tokens
    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        chunk_str = encoder.decode(chunk_tokens)
        chunks.append(chunk_str)
        start += chunk_size - overlap
    return chunks


def index_pdfs(root_dir, output_path):
    """Walk the directory, extract and embed text chunks, and save to disk."""
    docs = []
    print(f"Indexing PDFs under {root_dir}...")
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.pdf'):
                continue
            full_path = os.path.join(dirpath, fname)
            # Read PDF and split by page
            try:
                reader = PdfReader(full_path)
            except PdfReadError as e:
                print(f"⚠️ Skipping unreadable PDF: {full_path} ({e})")
                continue
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ''
                except Exception:
                    page_text = ''
                if not page_text.strip():
                    continue
                # Chunk this page's text
                for chunk_idx, chunk in enumerate(chunk_text(page_text)):
                    docs.append({
                        'path': full_path,
                        'page': page_num + 1,
                        'chunk': chunk_idx,
                        'text': chunk
                    })
    print(f"Created {len(docs)} chunks.")
    # Embed all chunks
    embeddings = []
    for idx, doc in enumerate(docs, 1):
        print(f"Embedding chunk {idx}/{len(docs)}...", end='\r')
        try:
            resp = openai.Embedding.create(
                input=doc['text'],
                model='text-embedding-ada-002'
            )
            emb = resp['data'][0]['embedding']
        except Exception as e:
            print(f"\nError embedding chunk {idx}: {e}", file=sys.stderr)
            emb = [0.0] * 1536
        embeddings.append(emb)
    # Convert to array and normalize for cosine (inner product) search
    embeddings = np.array(embeddings, dtype=np.float32)
    # In-place L2-normalize rows
    faiss.normalize_L2(embeddings)
    # Build FAISS index for inner-product search
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    # Determine output filenames
    base, ext = os.path.splitext(output_path)
    docs_file = base + '.pkl'
    faiss_file = base + '.faiss'
    # Save documents metadata
    with open(docs_file, 'wb') as f:
        pickle.dump(docs, f)
    # Save FAISS index
    faiss.write_index(index, faiss_file)
    print(f"\nSaved {len(docs)} document chunks to {docs_file}")
    print(f"Saved FAISS index to {faiss_file}")

def answer_question(docs, faiss_index, question, top_k=5, temperature=0.2):
    """Embed question, retrieve top_k contexts, and return the model's answer."""
    print("Embedding question...")
    resp = openai.Embedding.create(
        input=question,
        model=EMBED_MODEL
    )
    q_emb = np.array(resp['data'][0]['embedding'], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    # Perform a broader search then select top_k unique documents
    n_total = faiss_index.ntotal
    k_search = min(n_total, top_k * 5)
    distances, indices = faiss_index.search(q_emb, k_search)
    # Filter for unique document paths
    selected = []
    seen_paths = set()
    for idx in indices[0]:
        entry = docs[idx]
        path = entry.get('path')
        if path not in seen_paths:
            seen_paths.add(path)
            selected.append(entry)
            if len(selected) >= top_k:
                break
    # If not enough unique, fill with next best
    if len(selected) < top_k:
        for idx in indices[0]:
            entry = docs[idx]
            if entry not in selected:
                selected.append(entry)
                if len(selected) >= top_k:
                    break
    # Extract texts for prompt
    context_texts = [entry['text'] for entry in selected]
    system_prompt = "You are a helpful assistant that answers questions based on provided document excerpts."
    user_prompt = "Context:\n" + "\n---\n".join(context_texts) + f"\nQuestion: {question}"
    print("Querying chat completion...")
    chat_resp = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=temperature
    )
    answer = chat_resp['choices'][0]['message']['content']
    return answer, selected


def query_index(index_path, question, top_k=5):
    """Load index, retrieve relevant chunks, and ask the LLM."""
    # Determine filenames
    base, ext = os.path.splitext(index_path)
    docs_file = base + '.pkl'
    faiss_file = base + '.faiss'
    # Load documents metadata
    with open(docs_file, 'rb') as f:
        docs = pickle.load(f)
    # Load FAISS index
    index = faiss.read_index(faiss_file)
    # Retrieve answer and references using FAISS and ChatCompletion
    answer, contexts = answer_question(docs, index, question, top_k)
    print("\nAnswer:\n")
    print(answer)
    # Print references
    print("\nReferences:")
    for idx, ctx in enumerate(contexts, start=1):
        path = ctx.get('path', '<unknown>')
        page = ctx.get('page', 'N/A')
        print(f"[{idx}] {path} (page {page})")


def repl_chat(index_prefix, top_k=5, temperature=0.2):
    """Interactive REPL mode for querying PDF index."""
    # Load documents and FAISS index
    base, ext = os.path.splitext(index_prefix)
    docs_file = base + '.pkl'
    faiss_file = base + '.faiss'
    with open(docs_file, 'rb') as f:
        docs = pickle.load(f)
    index = faiss.read_index(faiss_file)
    print("Entering interactive REPL mode. Type 'exit' or press Ctrl-D to quit.")
    # REPL loop
    while True:
        try:
            question = input("\nUser> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL.")
            break
        if not question or question.strip().lower() in ('exit', 'quit'):
            print("Exiting REPL.")
            break
        # Get answer and contexts
        answer, contexts = answer_question(docs, index, question, top_k, temperature)
        print(f"\nAssistant> {answer}")
        # Print references
        print("References:")
        for idx, ctx in enumerate(contexts, start=1):
            path = ctx.get('path', '<unknown>')
            page = ctx.get('page', 'N/A')
            print(f" [{idx}] {path} (page {page})")

def main():
    parser = argparse.ArgumentParser(
        prog='chatpdf',
        description="Chat with your local PDF collection."
    )
    parser.add_argument('--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(dest='command')

    parser_index = subparsers.add_parser('index', help='Index PDFs under a directory')
    parser_index.add_argument('root_dir', help='Root directory to search for PDFs')
    parser_index.add_argument('index_prefix', help='Prefix for output index files (without extension)')

    parser_query = subparsers.add_parser('query', help='Query the indexed PDFs')
    parser_query.add_argument('index_prefix', help='Prefix of the index files (without extension)')
    parser_query.add_argument('question', nargs='+', help='Question to ask')
    parser_query.add_argument('--top_k', type=int, default=5, help='Number of top chunks to use')

    parser_repl = subparsers.add_parser('repl', help='Interactive REPL mode')
    parser_repl.add_argument('index_prefix', help='Prefix of the index files (without extension)')
    parser_repl.add_argument('--top_k', type=int, default=5, help='Number of top chunks to use')
    parser_repl.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature for chat model')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    load_api_key()
    if args.command == 'index':
        index_pdfs(args.root_dir, args.index_prefix)
    elif args.command == 'query':
        question = ' '.join(args.question)
        query_index(args.index_prefix, question, args.top_k)
    elif args.command == 'repl':
        repl_chat(args.index_prefix, args.top_k, args.temperature)


if __name__ == '__main__':
    main()
