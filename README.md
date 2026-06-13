# PaperMind

PaperMind 是一个基于 LangChain 和 LangGraph 构建的本地论文知识库 RAG Agent。

## 项目目标

面向研究生和科研人员本地论文难以统一检索、阅读后难以快速回顾的问题，将本地 PDF 论文构建为可检索的个人知识库。

系统计划支持：

- 本地 PDF 论文解析与分块
- 论文向量化与持久化存储
- 单篇论文和全论文库问答
- Dense 与 BM25 混合检索
- 带论文名称、页码和原文片段的可追溯回答
- 基于 LangGraph 的检索评估与问题改写流程
- 多轮对话与 Checkpoint 状态持久化

## 开发计划

### V1：基础论文 RAG

- PDF 解析
- 文本分块
- Embedding
- Chroma 向量存储
- 基础检索问答
- 引用来源展示

### V2：检索质量优化

- Dense + BM25 混合检索
- RRF 结果融合
- 检索相关性评估
- 问题改写与失败兜底

### V3：Agent 工作流

- LangGraph State
- 条件路由
- 多轮对话
- 消息窗口
- SQLite Checkpointer

## 当前进度

- [x] 项目初始化
- [ ] PDF 文档解析
- [ ] 文本分块
- [ ] 向量化入库
- [ ] 基础检索
- [ ] RAG 回答生成
EOF