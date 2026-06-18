from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "papermind_papers"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def create_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def search(query: str, top_k: int = 5) -> None:
    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    results = vector_store.similarity_search(
        query=query,
        k=top_k,
    )

    if not results:
        print("没有检索到结果，请确认PDF已经成功入库。")
        return

    print(f"\n查询：{query}")
    print(f"返回结果：{len(results)} 条")

    for index, document in enumerate(results, start=1):
        metadata = document.metadata

        print("\n" + "=" * 70)
        print(f"结果 {index}")
        print(f"文件：{metadata.get('file_name')}")
        print(f"页码：{metadata.get('page_number')}")
        print(f"章节：{metadata.get('section') or '未识别'}")
        print(f"Chunk：{metadata.get('chunk_index')}")
        print("-" * 70)
        print(document.page_content[:1000])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从PaperMind的Chroma知识库中检索论文内容。"
    )

    parser.add_argument(
        "query",
        type=str,
        help="检索问题",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回结果数量，默认5",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    search(args.query, args.top_k)