#!/usr/bin/env python3
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


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def index_pdfs(root_dir, output_path):
    """Walk the directory, extract and embed text chunks, and save to disk."""
    docs = []
    print(f"Indexing PDFs under {root_dir}...")
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.pdf'):
                full_path = os.path.join(dirpath, fname)
                text = extract_text_from_pdf(full_path)
                if not text:
                    continue
                for i, chunk in enumerate(chunk_text(text)):
                    docs.append({
                        'path': full_path,
                        'chunk_index': i,
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
    # Embed and normalize question
    print("Embedding question...")
    resp = openai.Embedding.create(
        input=question,
        model='text-embedding-ada-002'
    )
    q_emb = np.array(resp['data'][0]['embedding'], dtype=np.float32)
    # Normalize query vector for inner-product search
    q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    # Search for nearest neighbors
    distances, indices = index.search(q_emb, top_k)
    context = [docs[i]['text'] for i in indices[0]]
    # Build prompt
    system_prompt = (
        "You are a helpful assistant that answers questions based on provided document excerpts."
    )
    user_prompt = (
        "Context:\n" + "\n---\n".join(context) + f"\nQuestion: {question}"
    )
    # Ask ChatGPT
    print("Querying chat completion...")
    chat_resp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
    )
    answer = chat_resp['choices'][0]['message']['content']
    print("\nAnswer:\n")
    print(answer)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with your local PDF collection."
    )
    subparsers = parser.add_subparsers(dest='command')

    parser_index = subparsers.add_parser('index', help='Index PDFs under a directory')
    parser_index.add_argument('root_dir', help='Root directory to search for PDFs')
    parser_index.add_argument('index_path', help='Path to save the index file (e.g., index.pkl)')

    parser_query = subparsers.add_parser('query', help='Query the indexed PDFs')
    parser_query.add_argument('index_path', help='Path to the index file (e.g., index.pkl)')
    parser_query.add_argument('question', nargs='+', help='Question to ask')
    parser_query.add_argument('--top_k', type=int, default=5, help='Number of top chunks to use')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    load_api_key()
    if args.command == 'index':
        index_pdfs(args.root_dir, args.index_path)
    elif args.command == 'query':
        question = ' '.join(args.question)
        query_index(args.index_path, question, args.top_k)


if __name__ == '__main__':
    main()
