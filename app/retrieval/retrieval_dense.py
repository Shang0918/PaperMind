from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """统一的检索结果结构。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    rank: int
    section: str 
    source: str = "dense"


    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def file_name(self) -> str:
        return str(self.metadata.get("file_name", ""))

    @property
    def page_number(self) -> int | None:
        value = self.metadata.get("page_number")

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def chunk_index(self) -> int | None:
        value = self.metadata.get("chunk_index")

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class DenseRetriever:
    """基于 Chroma 的 Dense 向量检索器。"""

    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store

    def search_global(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        全库检索。

        适用问题：
        - 哪篇论文使用图注意力网络？
        - 哪些方法使用了作者关系？
        """
        self._validate_query(query)
        self._validate_top_k(top_k)

        docs_with_scores = (
            self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
            )
        )

        return self._convert_results(docs_with_scores)

    def search_single_document(
        self,
        query: str,
        doc_id: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        单论文检索。

        只在指定 doc_id 对应的论文 Chunk 中进行向量排序。
        """
        self._validate_query(query)
        self._validate_doc_id(doc_id)
        self._validate_top_k(top_k)

        docs_with_scores = (
            self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter={
                    "doc_id": doc_id,
                },
            )
        )

        return self._convert_results(docs_with_scores)

    def search_multiple_documents(
        self,
        query: str,
        doc_ids: list[str],
        top_k_per_doc: int = 5,
    ) -> dict[str, list[RetrievalResult]]:
        """
        多论文检索。

        对每一篇论文分别检索，保证每篇目标论文都有自己的 Top-K，
        避免某一篇论文占满全部返回结果。
        """
        self._validate_query(query)
        self._validate_top_k(top_k_per_doc)

        normalized_doc_ids = self._normalize_doc_ids(doc_ids)

        if len(normalized_doc_ids) < 2:
            raise ValueError(
                "多论文检索至少需要两个不同的 doc_id。"
            )

        results_by_document: dict[
            str,
            list[RetrievalResult],
        ] = {}

        for doc_id in normalized_doc_ids:
            results_by_document[doc_id] = (
                self.search_single_document(
                    query=query,
                    doc_id=doc_id,
                    top_k=top_k_per_doc,
                )
            )

        return results_by_document

    @staticmethod
    def _convert_results(
        docs_with_scores: list[
            tuple[Document, float]
        ],
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []

        for rank, (document, distance) in enumerate(
            docs_with_scores,
            start=1,
        ):
            metadata = dict(document.metadata)

            chunk_id = str(
                metadata.get("chunk_index", "")
            )


            section = str(
                metadata.get("section","")
            ).strip().lower()

            if not chunk_id:
                raise ValueError(
                    "检索结果缺少 chunk_id。"
                    "请确认入库时已将 chunk_id 写入 metadata。"
                )
            
            if  "references" in section:
                continue
        

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=document.page_content,
                    metadata=metadata,
                    distance=float(distance),
                    rank=rank,
                    section = section
                )
            )

        return results

    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not query.strip():
            raise ValueError("查询内容不能为空。")

    @staticmethod
    def _validate_doc_id(doc_id: str) -> None:
        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id 不能为空。")

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

    @staticmethod
    def _normalize_doc_ids(
        doc_ids: list[str],
    ) -> list[str]:
        if not doc_ids:
            raise ValueError("doc_ids 不能为空。")

        normalized: list[str] = []
        seen: set[str] = set()

        for doc_id in doc_ids:
            cleaned = str(doc_id).strip()

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            normalized.append(cleaned)

        return normalized