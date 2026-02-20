# run: python main.py index | chat | query "question"
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import INDEX_DIR, UPLOADS_DIR


def cmd_index(args):
    from indexer import run_indexing
    source = Path(args.source) if args.source else UPLOADS_DIR
    result = run_indexing(source=source, index_dir=INDEX_DIR)
    print(f"Indexed {result['n_docs']} docs, {result['n_chunks']} chunks. Status: {result['status']}")


def cmd_chat(args):
    import subprocess
    app = PROJECT_ROOT / "ui" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"], cwd=PROJECT_ROOT)


def cmd_query(args):
    from pipeline import run_pipeline
    out = run_pipeline(
        args.query,
        use_hybrid=not args.dense_only,
        use_query_rewrite=True,
        index_dir=INDEX_DIR,
    )
    if out.get("error"):
        print("Error:", out["error"])
        return
    print("Answer:", out["answer"])
    print(f"Confidence: {out['confidence']:.2f} | Retrieval: {out['retrieval_time_s']:.2f}s | Gen: {out['generation_time_s']:.2f}s")


def cmd_gpu_check(args):
    from config import USE_GPU, DEVICE, LLM_GPU_LAYERS
    print("=== GPU Status ===")
    print(f"GPU detected: {USE_GPU}")
    print(f"Device: {DEVICE}")
    print(f"LLM GPU layers: {LLM_GPU_LAYERS}")
    if USE_GPU:
        try:
            import torch
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
        except ImportError:
            print("PyTorch not installed - install with: pip install torch")


def main():
    parser = argparse.ArgumentParser(description="offline RAG system")
    sub = parser.add_subparsers(dest="command", required=True)
    
    p_index = sub.add_parser("index", help="Build index from documents")
    p_index.add_argument("--source", default=None, help=f"Directory or file (default: {UPLOADS_DIR})")
    p_index.set_defaults(run=cmd_index)
    
    p_chat = sub.add_parser("chat", help="Launch Streamlit UI")
    p_chat.set_defaults(run=cmd_chat)
    
    p_query = sub.add_parser("query", help="Single query (requires index)")
    p_query.add_argument("query", help="Question")
    p_query.add_argument("--dense-only", action="store_true", help="Use dense retrieval only")
    p_query.set_defaults(run=cmd_query)
    
    p_gpu = sub.add_parser("gpu-check", help="Check GPU availability")
    p_gpu.set_defaults(run=cmd_gpu_check)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
