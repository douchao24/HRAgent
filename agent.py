import os
from tkinter import constants
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.indexes import SQLRecordManager, index

# 1. 导入监听模块（核心！）
from watch_docs import start_file_monitor
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware


# 从环境变量读取 API Key（建议在系统环境变量中设置 OPENAI_API_KEY）
api_key = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxx")
if not api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

PERSIST_DIRECTORY = "./chromaDB"
NAMESPACE = "chroma/my_knowledge_base"
RECORD_DB_URL = "sqlite:///record_manager_cache.sql"

# 清理模式: "incremental" | "full" | "scoped_full" | None
#   incremental - 仅处理变动，不清理已删除的源文件（推荐日常使用）
#   full        - 处理变动 + 清理已删除的源文件（需传入全部文档）
CLEANUP_MODE = "incremental"

# 元数据中标识文档来源的字段名
SOURCE_ID_KEY = "source"


# ============================================================
# 基础组件
# ============================================================

def get_embeddings():
    """获取共享的嵌入模型实例"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"}
    )


def get_vectorstore():
    """获取或创建向量数据库实例（连接已有的，不会清空数据）"""
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIRECTORY
    )


def init_record_manager():
    """初始化记录管理器"""
    record_manager = SQLRecordManager(
        namespace=NAMESPACE,
        db_url=RECORD_DB_URL,
    )
    record_manager.create_schema()
    return record_manager


# ============================================================
# 全局文本分割器
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    length_function=len,
    add_start_index=True,
)


# ============================================================
# 文档加载与索引（核心修复区域）
# ============================================================
def load_and_split_pdf(pdf_path: str):
    """
    加载 PDF 文档并进行文本分块。
    返回分块后的文档列表，每个文档的 metadata 中包含 source 字段。
    """
    # 加载文档
    try:
        loader = UnstructuredLoader(
            file_path=pdf_path,
            extract_images=True,
            extract_tables=True,
            chunking_strategy="by_title"
        )
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"PDF加载失败: {e}")

    if not docs:
        raise ValueError("PDF文档为空或无法解析")

    # 确保 source 字段为绝对路径（RecordManager 依赖此字段追踪文档）
    abs_path = os.path.abspath(pdf_path)
    for doc in docs:
        doc.metadata["source"] = abs_path

    # 分割文本
    chunks = text_splitter.split_documents(docs)
    print(f"[分块] {len(docs)} 页 -> {len(chunks)} 个文本块")
    return chunks


def run_indexing(chunks: list, cleanup: str = CLEANUP_MODE) -> dict:
    """
    执行增量索引。

    ✅ 修复2 & 3: 不再调用 build_knowledge_base，直接使用 get_vectorstore()
    这样 index() 函数写入的向量库就是后续 RAG 使用的同一个。

    Args:
        chunks: 待索引的文档块列表
        cleanup: 清理模式
            - None: 仅去重，不清理
            - "incremental": 检测内容变动，自动更新（不处理源文件删除）
            - "full": 检测内容变动 + 清理已删除的源文件（需传入全部文档）
            - "scoped_full": 批量索引结束时统一清理变动

    Returns:
        索引结果统计字典
    """
    vectorstore = get_vectorstore()
    vectorstore.delete(where={"source": './docs/*.pdf'})
    record_manager = init_record_manager()

    print(f"\n{'='*60}")
    print(f"开始索引 | 清理模式: {cleanup} | 文档块数: {len(chunks)}")
    print(f"{'='*60}")

    result = index(
        chunks,
        record_manager,
        vectorstore,
        cleanup=cleanup,
        source_id_key=SOURCE_ID_KEY,
    )

    print(f"\n[索引结果]")
    print(f"  新增 (num_added):   {result['num_added']}")
    print(f"  更新 (num_updated): {result['num_updated']}")
    print(f"  跳过 (num_skipped): {result['num_skipped']}")
    print(f"  删除 (num_deleted): {result['num_deleted']}")

    return result


def build_knowledge_base(pdf_path: str):
    """
    完整的知识库构建流程：加载 -> 分块 -> 增量索引。
    ✅ 修复: 不再创建新的向量库，而是复用 get_vectorstore()。
    """
    chunks = load_and_split_pdf(pdf_path)
    run_indexing(chunks, cleanup=CLEANUP_MODE)
    return get_vectorstore()


observer = start_file_monitor(init_record_manager(), get_vectorstore(), text_splitter, watch_path="docs/")

# ============================================================
# RAG 链
# ============================================================

def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        model='qwen3.6-plus',
        api_key=api_key,
        # ✅ 修复4: 去掉 base_url 中的反引号
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1
    )


def setup_rag_chain(llm=None):
    """设置 RAG 链"""
    if llm is None:
        llm = create_llm()

    vectorstore = get_vectorstore()

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

    # 定义提示词模板
    template = """你是一个知识渊博的AI助手。请严格依据以下【上下文】中的信息，回答用户的问题。
    如果【上下文】中没有提供足够信息，请如实告知用户。

    【上下文】:
    {context}

    用户问题: {question}
    回答:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        """将检索到的文档列表格式化为字符串"""
        return "\n\n".join([d.page_content for d in docs])

    # 构建 RAG 链 (LCEL语法)
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} |
        prompt |
        llm |
        StrOutputParser()
    )

    return rag_chain


# ============================================================
# 主程序入口
# ============================================================

app = FastAPI(title="RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/chat")
def chat(question: str = Query(...)):
    try:
      rag_chain = setup_rag_chain()
      response = rag_chain.invoke(question)
      answer = str(response)
      return {
        "code": 200,
        "question": question,
        "answer": answer
      }
    except Exception as e:
        return {"code": 500, "error": str(e)}


# 刷新重建知识库
@app.get("/refresh")
def refresh(pdf_name: str = Query(..., description="你的PDF文件名")):
    try:
        pdf_path = os.path.join("docs", pdf_name)
        if not os.path.exists(pdf_path):
            return {"code": 404, "error": f"文件不存在: {pdf_path}"}
        
        build_knowledge_base(pdf_path)
        return {"code": 200, "msg": "知识库重建成功"}
    except Exception as e:
        return {"code": 500, "error": str(e)}