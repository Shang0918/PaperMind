#读取PDF 切分入库
import os
from pathlib import Path
import hashlib
import re 
import torch
from langchain_core.documents import Document
import pymupdf4llm
from langchain_text_splitters import(
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "papermind_papers"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","BAAI/bge-m3")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32


def calculate_file_hash(file_path:Path) -> str:
    """
    根据PDF内容生成文档ID

    同一个 PDF 即使路径变化，doc_id 也能保持一致
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        while block := file.read(1024*1024):
            sha256.update(block)
    
    return sha256.hexdigest()[:24]

def clean_markdown(text:str) -> str:
    """
    对PDF解析得到的Markdown做清洗
    """
    text = text.replace("\r\n","\n").replace("\r","\n")

    text = "\n".join(line.rstrip() for line in text.splitlines())

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_heading(heading: str | None) -> str:
    """
    清除标题中的Markdown格式字符。
    """
    if not heading:
        return ""

    heading = re.sub(r"[*_`#]", "", str(heading))
    heading = re.sub(r"\s+", " ", heading)

    return heading.strip()


"""
处理跳页标题问题
"""

HEADER_KEYS = [
    "header_1",
    "header_2",
    "header_3",
    "header_4",
]


def create_empty_heading_state() -> dict[str, str]:
    """
    创建一个空的标题层级状态。
    """
    return {
        "header_1": "",
        "header_2": "",
        "header_3": "",
        "header_4": "",
    }


def update_heading_state(
    heading_state: dict[str, str],
    current_metadata: dict,
) -> dict[str, str]:
    """
    使用当前章节检测到的标题更新跨页标题状态。

    规则：
    - 新一级标题出现：更新一级标题，清空二、三、四级标题
    - 新二级标题出现：保留一级标题，更新二级标题，清空三级和四级
    - 新三级标题出现：保留一级、二级标题，更新三级，清空四级
    - 当前段落没有标题：完整继承上一段标题状态
    """
    new_state = heading_state.copy()

    for level in range(1, 5):
        key = f"header_{level}"

        heading = clean_heading(
            current_metadata.get(key)
        )

        if not heading:
            continue

        new_state[key] = heading

        # 出现新的上级标题后，旧的下级标题全部失效
        for lower_level in range(level + 1, 5):
            lower_key = f"header_{lower_level}"
            new_state[lower_key] = ""

    return new_state


def build_section_path(
    heading_state: dict[str, str],
) -> str:
    """
    根据当前完整标题状态生成章节路径。
    """
    headings = [
        heading_state.get("header_1", ""),
        heading_state.get("header_2", ""),
        heading_state.get("header_3", ""),
        heading_state.get("header_4", ""),
    ]

    section_path = " > ".join(
        heading
        for heading in headings
        if heading
    )

    return section_path or "Front Matter"


def load_pdf_pages(pdf_path:Path) -> list[Document]:
    """
    将PDF解析成逐页Markdown文档
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在:{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"目前只支持PDF文件：{pdf_path}")
    
    doc_id = calculate_file_hash(pdf_path)

    print(f"正在解析PDF：{pdf_path.name}")

    page_results = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks = True,
        use_ocr = False,
        header=False,
        footer=False,
        show_progress=True,
    )

    if not isinstance(page_results,list):
        raise RuntimeError("PDF解析结果异常")
    
    page_documents: list[Document] = []
    total_text_length = 0

    for index,page_result in enumerate(page_results,start=1):
        text = clean_markdown(page_result.get("text",""))

        if len(text)<20:
            continue

        raw_metadata = page_result.get("metadata",{})
        page_number = int(raw_metadata.get("page_number",index))

        metadata={
            "doc_id":doc_id,
            "file_name":pdf_path.name,
            "source_path":str(pdf_path.resolve()),
            "page_number":page_number,
            "page_count":len(page_results),
            "document_title":raw_metadata.get("title") or pdf_path.stem,
        }

        page_documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

        total_text_length += len(text)

    if total_text_length < 200:
        raise ValueError(
            "PDF中没有提取到足够文本。"
            "该文件可能是扫描版PDF，当前版本暂未启用OCR。"
        )
    
    print(
        f"PDF解析完成：有效页数={len(page_documents)}，"
        f"文本字符数={total_text_length}"
    )

    return page_documents

def calculate_effective_length(text: str) -> int:
    """
    计算Chunk中有效正文的字符数量。
    """
    cleaned = text

    # 删除Markdown标题
    cleaned = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s+.*$",
        "",
        cleaned,
    )

    # 删除Markdown图片
    cleaned = re.sub(
        r"!\[[^\]]*]\([^)]*\)",
        "",
        cleaned,
    )

    # 链接保留显示文字
    cleaned = re.sub(
        r"\[([^\]]+)]\([^)]*\)",
        r"\1",
        cleaned,
    )

    # 删除常见Markdown控制符
    cleaned = re.sub(
        r"[`*_>#|~\-]",
        "",
        cleaned,
    )

    effective_chars = re.findall(
        r"[\w\u4e00-\u9fff]",
        cleaned,
        flags=re.UNICODE,
    )

    return len(effective_chars)


def build_retrieval_content(
    body: str,
    metadata: dict,
) -> str:
    """
    不将 元数据 导入 文本中
    """
    document_title = metadata.get("document_title")

    if not document_title:
        return body.strip()
    
    return(
        f"Paper:{document_title}\n\n"
        f"{body.strip()}"
    )

def split_page_documents(
    page_documents: list[Document],
) -> list[Document]:
    """
    PDF切分流程：

    1. 按页码顺序处理。
    2. 使用Markdown标题切分每页。
    3. 使用heading_state保存跨页标题层级。
    4. 没有标题的内容继承上一段标题。
    5. 新标题只更新对应层级，并清除失效的下级标题。
    6. 对章节正文继续进行递归切分。
    7. 删除低信息Chunk。
    """

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ],
        strip_headers=True,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            "。", "！", "？",
            ". ", "! ", "? ",
            "；", "; ",
            "，", ", ",
            " ",
            "",
        ],
    )

    # 防止页面顺序异常
    sorted_page_documents = sorted(
        page_documents,
        key=lambda document: document.metadata.get(
            "page_number",
            0,
        ),
    )

    candidate_chunks: list[Document] = []

    # 关键：这个状态在整篇PDF处理期间持续存在
    heading_state = create_empty_heading_state()

    for page_document in sorted_page_documents:
        section_documents = header_splitter.split_text(
            page_document.page_content
        )

        # 如果这一页经过标题切分后没有正文，
        # 通常意味着该页只有标题、图片或空白内容。
        # 不要再把原始页面重新作为正文，否则会重新产生标题Chunk。
        if not section_documents:
            continue

        prepared_sections: list[Document] = []

        for section_document in section_documents:
            # 当前section的metadata只表示当前页面内
            # MarkdownHeaderTextSplitter识别出的标题
            current_header_metadata = (
                section_document.metadata
            )

            # 将当前标题合并到跨页状态中
            heading_state = update_heading_state(
                heading_state=heading_state,
                current_metadata=current_header_metadata,
            )

            # 使用完整跨页状态构建章节路径
            section_path = build_section_path(
                heading_state
            )

            body = section_document.page_content.strip()

            if not body:
                continue

            metadata = {
                **page_document.metadata,

                # 保存每一级标题，后续方便元数据过滤
                "header_1": heading_state["header_1"],
                "header_2": heading_state["header_2"],
                "header_3": heading_state["header_3"],
                "header_4": heading_state["header_4"],

                # 保存完整章节路径
                "section": section_path,
            }

            prepared_sections.append(
                Document(
                    page_content=body,
                    metadata=metadata,
                )
            )

        # 对当前页的各章节正文进一步切分
        page_chunks = recursive_splitter.split_documents(
            prepared_sections
        )

        candidate_chunks.extend(page_chunks)

    # ==========================
    # 过滤低信息Chunk
    # ==========================

    valid_chunks: list[Document] = []
    removed_chunks: list[Document] = []

    for chunk in candidate_chunks:
        effective_length = calculate_effective_length(
            chunk.page_content
        )

        if effective_length < 40:
            removed_chunks.append(chunk)
            continue

        chunk.metadata["body_char_count"] = len(
            chunk.page_content
        )

        chunk.metadata["effective_char_count"] = (
            effective_length
        )

        chunk.page_content = build_retrieval_content(
            body=chunk.page_content,
            metadata=chunk.metadata,
        )

        valid_chunks.append(chunk)

    # 添加最终Chunk序号
    for chunk_index, chunk in enumerate(valid_chunks):
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["char_count"] = len(
            chunk.page_content
        )

    if not valid_chunks:
        raise ValueError(
            "切分完成后没有有效Chunk，请检查PDF解析结果。"
        )

    average_length = (
        sum(
            len(chunk.page_content)
            for chunk in valid_chunks
        )
        // len(valid_chunks)
    )

    print(f"初始Chunk数量：{len(candidate_chunks)}")
    print(f"删除低信息Chunk：{len(removed_chunks)}")
    print(f"最终Chunk数量：{len(valid_chunks)}")
    print(f"平均Chunk长度：{average_length}")

    if removed_chunks:
        print("\n被过滤的Chunk示例：")

        for chunk in removed_chunks[:5]:
            print("-" * 50)
            print(repr(chunk.page_content[:200]))
            print(chunk.metadata)

    return valid_chunks

def create_embeddings() -> HuggingFaceEmbeddings:
    """
    创建本地Embedding模型。
    """
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    print(f"Embedding模型：{EMBEDDING_MODEL}")
    print(f"Embedding设备：{device}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": BATCH_SIZE,
        },
    )

def create_vector_store(
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """
    创建或连接本地持久化Chroma数据库。
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

def create_chunk_id(chunk: Document) -> str:
    """
    为每个chunk创建稳定ID。

    使用doc_id、页码、chunk序号和文本内容共同生成，
    避免同一个PDF重复运行时产生完全重复的数据。
    """
    identity = "|".join(
        [
            str(chunk.metadata["doc_id"]),
            str(chunk.metadata["page_number"]),
            str(chunk.metadata["chunk_index"]),
            chunk.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

def store_chunks(
    chunks: list[Document],
    vector_store: Chroma,
) -> None:
    """
    分批写入Chroma，避免一次向量化过多文本。
    """
    chunk_ids = [create_chunk_id(chunk) for chunk in chunks]

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE

        batch_documents = chunks[start:end]
        batch_ids = chunk_ids[start:end]

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        print(
            f"已写入：{min(end, len(chunks))}/{len(chunks)}"
        )

    print("Chroma写入完成。")

def print_chunk_examples(chunks: list[Document], count: int = 3) -> None:
    """
    打印部分chunk，人工检查切分质量。
    """
    print("\n========== Chunk示例 ==========")

    for index, chunk in enumerate(chunks[:count], start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"metadata: {chunk.metadata}")
        print(chunk.page_content[:500])

def ingest_pdf(pdf_path: Path) -> None:
    page_documents = load_pdf_pages(pdf_path)
    chunks = split_page_documents(page_documents)

    print_chunk_examples(chunks)

    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)

    store_chunks(chunks, vector_store)

    print("\n========== 入库结果 ==========")
    print(f"文件：{pdf_path.name}")
    print(f"页数：{len(page_documents)}")
    print(f"Chunks：{len(chunks)}")
    print(f"Collection：{COLLECTION_NAME}")
    print(f"Chroma路径：{CHROMA_DIR}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将学术PDF解析、切分并存入Chroma。"
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="待处理PDF文件路径",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()

    # try:
    #     ingest_pdf(args.pdf_path)
    #     # ingest_pdf("/private/shang/papermind/data/papers/DCNN.pdf")
    # except Exception as error:
    #     print(f"\n处理失败：{error}")
    #     raise

    folder_path = PROJECT_ROOT / "data" / "papers"
    file_paths = folder_path.glob("*.pdf")
    for file in file_paths:
        ingest_pdf(file) 

