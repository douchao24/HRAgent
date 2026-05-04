# 知识库智能问答系统 (Knowledge Base RAG System)

一个基于 LangChain 的智能问答系统，使用 RAG (Retrieval-Augmented Generation) 技术，支持文档自动监听和增量更新。

## 功能特点

- 📚 **多格式文档支持**: 支持 PDF、DOCX、TXT、MD 等文档格式
- 🔍 **智能检索**: 基于向量数据库的语义检索
- 🤖 **RAG 问答**: 结合检索上下文的智能回答
- 👁️ **文件监听**: 自动检测 docs 目录下的文件变化并实时更新知识库
- 📈 **增量索引**: 高效处理文档更新，避免重复索引
- 🎯 **中文优化**: 使用 BGE 中文嵌入模型，优化中文理解

## 技术栈

- **LangChain**: LLM 应用开发框架
- **ChromaDB**: 向量数据库
- **OpenAI API**: 兼容 DashScope (阿里云千问)
- **HuggingFace Embeddings**: BGE 中文嵌入模型
- **Watchdog**: 文件系统监听

## 安装

### 环境要求

- Python 3.8+
- Git

### 安装步骤

1. 克隆或下载项目
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

### API Key 配置

在 `agent.py` 第 18 行设置您的 API Key：

```python
api_key = "your-api-key-here"
```

或者设置环境变量：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here
```

### 模型配置

默认使用阿里云 DashScope 的 qwen3.6-plus 模型，如需修改，请编辑 `agent.py` 第 173 行：

```python
model='qwen3.6-plus'
```

## 使用方法

### 1. 准备文档

将您的文档放入 `docs/` 目录下。系统默认支持：
- `.pdf` - PDF 文档
- `.docx` - Word 文档
- `.txt` - 文本文件
- `.md` - Markdown 文件

### 2. 运行程序

```bash
python agent.py
```

### 3. 开始问答

程序启动后，输入您的问题即可获得基于知识库的回答。输入 `exit` 退出程序。

## 目录结构


## 核心功能说明

### 增量索引

系统使用增量索引策略，支持三种清理模式：

- `incremental` (默认): 仅处理变动，不清理已删除文档
- `full`: 处理变动 + 清理已删除文档
- `scoped_full`: 批量索引结束时统一清理

### 文件监听

程序启动后会自动监听 `docs/` 目录：
- 新增文件 → 自动添加到知识库
- 修改文件 → 自动更新索引
- 删除文件 → 可通过 full 模式清理

## 自定义配置

### 向量数据库路径

在 `agent.py` 第 22 行修改：

```python
PERSIST_DIRECTORY = "./chromaDB"
```

### 文本分块参数

在 `agent.py` 第 68-74 行修改：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # 分块大小
    chunk_overlap=100,     # 重叠大小
    separators=[...],      # 分隔符
)
```

## 常见问题

### Q: 如何更换嵌入模型？

A: 修改 `agent.py` 第 39-44 行的 `get_embeddings()` 函数。

### Q: 如何调整检索结果数量？

A: 修改 `agent.py` 第 189 行的 `search_kwargs={'k': 6}`。

## 许可证
MIT License
