import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# 从环境变量读取 API Key（建议在系统环境变量中设置 OPENAI_API_KEY）
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxx"
if not api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

PERSIST_DIRECTORY = "./chroma_db"


def get_embeddings():
    """获取共享的嵌入模型实例"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"}
    )

# --- 2. 构建知识库 (离线步骤，若已有则跳过) ---
def build_knowledge_base(pdf_path):
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

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)

    # 初始化嵌入模型和向量数据库
    # 新版 Chroma 会自动持久化，无需手动调用 persist()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

# --- 3. 检索器与RAG链定义 ---

def create_llm():
    """创建 LLM 实例，便于测试和配置管理"""
    return ChatOpenAI(
        model='qwen3.6-plus',
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1
    )


def setup_rag_chain(llm=None):
    """设置 RAG 链，支持注入 LLM 便于测试"""
    if llm is None:
        llm = create_llm()
    
    # 加载已经存在的向量数据库
    try:
        vectorstore = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIRECTORY
        )
    except Exception as e:
        raise RuntimeError(f"向量数据库加载失败: {e}")
    
    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

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
    # 使用 RunnableLambda 确保格式函数在链中正确执行
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} |
        prompt |
        llm |
        StrOutputParser()
    )

    return rag_chain



# --- 4. 主程序入口: 执行问答 ---
if __name__ == "__main__":
    # 检查是否需要构建知识库 (根据文件或文件夹的存在性判断)
    if not os.path.exists(PERSIST_DIRECTORY):
        print("正在构建知识库，请确保提供PDF文件路径...")
        # 请替换为你的PDF文件路径
        vectorstore = build_knowledge_base("./电商客服手册_含图片版.pdf")
    else:
        print("检测到已有知识库，正在加载...")
    
    try:
        rag_chain = setup_rag_chain()
    except RuntimeError as e:
        print(f"错误: {e}")
        exit(1)
    
    while True:
        user_question = input("\n请输入问题 (或输入 'exit' 退出): ")
        if user_question.lower() == 'exit':
            break
        try:
            # 调用RAG链生成答案
            response = rag_chain.invoke(user_question)
            print(f"\n回答: {response}")
        except Exception as e:
            print(f"生成回答时出错: {e}")