from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma

# 按你的实际位置修改这个导
from app.ingestion.ingest_pdf import (
    create_vector_store,
    create_embeddings,
)
from app.retrieval.retrieval_dense import (
    DenseRetriever,
    RetrievalResult,
)


def get_document_catalog(
    vector_store: Chroma,
) -> list[dict[str, Any]]:
    data = vector_store.get(
        include=["metadatas"],
    )

    metadatas = data.get("metadatas", [])
    documents: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:
        if not metadata:
            continue

        doc_id = str(
            metadata.get("doc_id", "")
        ).strip()

        if not doc_id:
            continue

        documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "file_name": metadata.get(
                    "file_name",
                    "",
                ),
                "document_title": metadata.get(
                    "document_title",
                    "",
                ),
            },
        )

    return list(documents.values())


def find_doc_id(
    catalog: list[dict[str, Any]],
    file_name: str,
) -> str:
    for document in catalog:
        if document["file_name"] == file_name:
            return document["doc_id"]

    raise ValueError(
        f"知识库中没有找到文件：{file_name}"
    )


def print_results(
    title: str,
    results: list[RetrievalResult],
) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    if not results:
        print("没有检索到结果。")
        return

    for result in results:
        preview = " ".join(
            result.text.split()
        )[:300]

        print(
            f"\n排名：{result.rank}"
            f"\n文件：{result.file_name}"
            f"\n页码：{result.page_number}"
            f"\nChunk：{result.chunk_index}"
            f"\n距离：{result.distance:.6f}"
            f"\n内容：{preview}"
            f"\n部分:{result.section}"
        )


def main() -> None:
    emb = create_embeddings()
    vector_store = create_vector_store(emb)
    retriever = DenseRetriever(vector_store)

    catalog = get_document_catalog(vector_store)

    print("当前论文目录：")

    for document in catalog:
        print(
            f"- {document['file_name']}: "
            f"{document['doc_id']}"
        )

    simcpsr_doc_id = find_doc_id(
        catalog,
        "SimCPSR.pdf",
    )

    graphconfrec_doc_id = find_doc_id(
        catalog,
        "GraphConfRec.pdf",
    )

    # 1. 单论文检索
    single_results = (
        retriever.search_single_document(
            query="这篇论文使用什么预训练语言模型？",
            doc_id=simcpsr_doc_id,
            top_k=5,
        )
    )

    print_results(
        "单论文检索：SimCPSR",
        single_results,
    )

    # 2. 全库检索
    global_results = retriever.search_global(
        query="哪篇论文使用图注意力网络进行会议推荐？",
        top_k=10,
    )

    print_results(
        "全库检索",
        global_results,
    )

    # 3. 多论文检索
    multi_results = (
        retriever.search_multiple_documents(
            query=(
                "这些方法如何利用论文之外的信息"
                "改进投稿推荐？"
            ),
            doc_ids=[
                simcpsr_doc_id,
                graphconfrec_doc_id,
            ],
            top_k_per_doc=5,
        )
    )

    for doc_id, results in multi_results.items():
        print_results(
            f"多论文检索，doc_id={doc_id}",
            results,
        )


if __name__ == "__main__":
    main()