#!/usr/bin/env python3
"""
Command-line tool to chat with a collection of local PDF documents using OpenAI APIs.
"""

__version__ = '0.1.0'

import os
import sys
import argparse
import pickle
import requests
from lxml import etree
# Optional semantic tokenizer (only required if --semantic is used)
try:
    from unstructured.partition.text import partition_text
except ImportError:
    partition_text = None

from dotenv import load_dotenv
import getpass
import json
import numpy as np
import faiss
import tiktoken

# OpenAI client and PDF reader
import openai
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

# Load user configuration
def load_config():
    """Load configuration from local .chatpdf.json and global ~/.config/chatpdf/config.json."""
    config = {}
    # Local config in cwd
    local_cfg = os.path.join(os.getcwd(), '.chatpdf.json')
    if os.path.exists(local_cfg):
        try:
            with open(local_cfg) as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Warning loading local config {local_cfg}: {e}", file=sys.stderr)
    # Global config under XDG_CONFIG_HOME or ~/.config/chatpdf/config.json
    home = os.path.expanduser('~')
    config_home = os.environ.get('XDG_CONFIG_HOME', os.path.join(home, '.config'))
    global_cfg = os.path.join(config_home, 'chatpdf', 'config.json')
    if os.path.exists(global_cfg):
        try:
            with open(global_cfg) as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Warning loading global config {global_cfg}: {e}", file=sys.stderr)
    return config

# Initialize config and model names
_CONFIG = load_config()
EMBED_MODEL = _CONFIG.get('embedding_model', 'text-embedding-ada-002')
CHAT_MODEL = _CONFIG.get('chat_model', 'gpt-3.5-turbo')
# Token-based chunking parameters
CHUNK_SIZE = _CONFIG.get('chunk_size', 500)
CHUNK_OVERLAP = _CONFIG.get('chunk_overlap', 50)


def load_api_key():
    # Load local .env first, then global config
    load_dotenv()
    # Load global config from XDG or ~/.config/chatpdf/.env
    home = os.path.expanduser("~")
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.join(home, ".config"))
    conf_file = os.path.join(config_home, "chatpdf", ".env")
    load_dotenv(conf_file)
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
 
def semantic_chunk_paragraphs(paragraphs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk list of paragraphs into semantic chunks based on token count."""
    # Initialize tokenizer
    try:
        encoder = tiktoken.encoding_for_model(CHAT_MODEL)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = []
    current_tokens = 0
    for para in paragraphs:
        tokens = encoder.encode(para)
        n_tokens = len(tokens)
        if current_chunk and current_tokens + n_tokens > chunk_size:
            chunks.append("\n\n".join(current_chunk))
            # overlap: keep last paragraph
            current_chunk = current_chunk[-1:]
            current_tokens = len(encoder.encode(current_chunk[0])) if current_chunk else 0
        current_chunk.append(para)
        current_tokens += n_tokens
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def extract_with_grobid(pdf_path, grobid_url):
    """Extract sections and paragraphs from PDF using Grobid fulltext API."""
    files = {'input': open(pdf_path, 'rb')}
    params = {'consolidateHeader': '0', 'consolidateCitations': '0'}
    try:
        resp = requests.post(f"{grobid_url}/api/processFulltextDocument", files=files, params=params)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error calling Grobid on {pdf_path}: {e}", file=sys.stderr)
        return []
    xml = resp.text
    try:
        root = etree.fromstring(xml.encode('utf-8'))
    except Exception as e:
        print(f"Error parsing Grobid XML for {pdf_path}: {e}", file=sys.stderr)
        return []
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    body = root.find('.//tei:text/tei:body', ns)
    if body is None:
        return []
    sections = []
    for div in body.findall('.//tei:div[@type="section"]', ns):
        head = div.find('tei:head', ns)
        title = head.text.strip() if head is not None and head.text else ''
        paras = []
        for p in div.findall('tei:p', ns):
            if p.text and p.text.strip():
                paras.append(p.text.strip())
        if paras:
            sections.append({'title': title, 'paragraphs': paras})
    return sections


def index_pdfs(root_dir, output_prefix, use_grobid=False, grobid_url=None, use_semantic=False):
    """Walk the directory, extract and embed text chunks (optionally via Grobid/Unstructured), and save index."""
    docs = []
    print(f"Indexing PDFs under {root_dir}...")
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.pdf'):
                continue
            full_path = os.path.join(dirpath, fname)
            # Extract content
            if use_grobid:
                sections = extract_with_grobid(full_path, grobid_url)
                for sec in sections:
                    paras = sec.get('paragraphs', [])
                    # Chunk by semantic or simple token window
                    if use_semantic:
                        chunks = semantic_chunk_paragraphs(paras)
                    else:
                        chunks = chunk_text("\n\n".join(paras))
                    for chunk_idx, chunk in enumerate(chunks):
                        docs.append({
                            'path': full_path,
                            'section': sec.get('title', ''),
                            'chunk': chunk_idx,
                            'text': chunk
                        })
            else:
                # Page-based or TXT fallback extraction
                text_pages = []
                try:
                    reader = PdfReader(full_path)
                    for page in reader.pages:
                        try:
                            page_text = page.extract_text() or ''
                        except Exception:
                            page_text = ''
                        if page_text.strip():
                            text_pages.append(page_text)
                except PdfReadError as e:
                    print(f"⚠️ Skipping unreadable PDF: {full_path} ({e})")
                    continue
                if not text_pages:
                    txt_path = os.path.splitext(full_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                txt_text = f.read()
                            if txt_text.strip():
                                text_pages = [txt_text]
                                print(f"ℹ️ Using TXT backup for {full_path}")
                        except Exception as e:
                            print(f"⚠️ Error reading TXT {txt_path}: {e}")
                # Skip if still no text
                if not text_pages:
                    continue
                for page_num, page_text in enumerate(text_pages, start=1):
                    # Semantic chunk per page if requested
                    if use_semantic:
                        paras = []
                        try:
                            elems = partition_text(text=page_text)
                            paras = [el.text.strip() for el in elems if getattr(el, 'element_type', '') == 'NarrativeText' and el.text]
                        except Exception:
                            paras = page_text.split('\n\n')
                        chunks = semantic_chunk_paragraphs(paras)
                    else:
                        chunks = chunk_text(page_text)
                    for chunk_idx, chunk in enumerate(chunks):
                        docs.append({
                            'path': full_path,
                            'page': page_num,
                            'chunk': chunk_idx,
                            'text': chunk
                        })
    print(f"Created {len(docs)} chunks.")
    # Report how many distinct PDFs were indexed
    unique_paths = set(doc['path'] for doc in docs)
    print(f"Indexed {len(unique_paths)} PDF files with extractable text.")
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
    base = output_prefix
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
    system_prompt = (
        "You are a helpful assistant that answers questions based on provided document excerpts. "
        "When including mathematical formulas or equations, format them using LaTeX notation—"
        "use $...$ for inline math and $$...$$ for display math."
    )
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


def auth_login():
    """Prompt for OpenAI API key and save to global config."""
    key = getpass.getpass("OpenAI API key: ")
    if not key:
        print("Error: no API key provided.", file=sys.stderr)
        sys.exit(1)
    # Write to XDG_CONFIG_HOME/chatpdf/.env
    home = os.path.expanduser("~")
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.join(home, ".config"))
    conf_dir = os.path.join(config_home, "chatpdf")
    os.makedirs(conf_dir, exist_ok=True)
    conf_file = os.path.join(conf_dir, ".env")
    try:
        with open(conf_file, 'w') as f:
            f.write(f"OPENAI_API_KEY={key}\n")
        print(f"Success: API key saved to {conf_file}")
    except Exception as e:
        print(f"Error saving API key: {e}", file=sys.stderr)
        sys.exit(1)


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

    # auth subcommand for API key setup
    parser_auth = subparsers.add_parser('auth', help='Manage OpenAI API key')
    auth_sub = parser_auth.add_subparsers(dest='auth_cmd')
    auth_sub.add_parser('login', help='Prompt and save your OpenAI API key')
    # index subcommand
    parser_index = subparsers.add_parser(
        'index',
        help='Index PDFs under a directory',
        epilog=(
            'Grobid setup (optional):\n'
            '  docker pull lfoppiano/grobid:0.7.3\n'
            '  docker run --rm -t -p 8070:8070 lfoppiano/grobid:0.7.3\n'
            '  export GROBID_URL=http://localhost:8070\n'
            'Use --grobid to enable Grobid-based extraction. See README for details.'
        )
    )
    parser_index.add_argument(
        'root_dir',
        help='Root directory to search for PDF files'
    )
    parser_index.add_argument(
        'index_prefix',
        help='Prefix for output index files (without extension)'
    )
    parser_index.add_argument(
        '--grobid', '-g',
        action='store_true',
        help='Enable Grobid-based section & metadata extraction (requires Grobid server). See README.'
    )
    parser_index.add_argument(
        '--grobid-url',
        default=os.getenv('GROBID_URL', 'http://localhost:8070'),
        help='URL of running Grobid service (default from GROBID_URL or http://localhost:8070)'
    )
    parser_index.add_argument(
        '--semantic', '-s',
        action='store_true',
        help='Enable Unstructured-based semantic chunking of paragraphs'
    )

    parser_query = subparsers.add_parser('query', help='Query the indexed PDFs')
    parser_query.add_argument('index_prefix', help='Prefix of the index files (without extension)')
    parser_query.add_argument('question', nargs='+', help='Question to ask')
    parser_query.add_argument('--top_k', type=int, default=5, help='Number of top chunks to use')

    parser_repl = subparsers.add_parser('repl', help='Interactive REPL mode')
    parser_repl.add_argument('index_prefix', help='Prefix of the index files (without extension)')
    parser_repl.add_argument('--top_k', type=int, default=5, help='Number of top chunks to use')
    parser_repl.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature for chat model')

    args = parser.parse_args()
    # Handle auth commands without requiring API key loaded
    if args.command == 'auth':
        if args.auth_cmd == 'login':
            auth_login()
        else:
            parser.print_help()
        sys.exit(0)
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load API key from env or config
    load_api_key()
    if args.command == 'index':
        index_pdfs(args.root_dir, args.index_prefix,
                   use_grobid=args.grobid,
                   grobid_url=args.grobid_url,
                   use_semantic=args.semantic)
    elif args.command == 'query':
        question = ' '.join(args.question)
        query_index(args.index_prefix, question, args.top_k)
    elif args.command == 'repl':
        repl_chat(args.index_prefix, args.top_k, args.temperature)


if __name__ == '__main__':
    main()
