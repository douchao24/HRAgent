# watch_docs.py 【最终完美版】
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_classic.indexes import index

# 1. 修复：替换废弃加载器
from langchain_unstructured import UnstructuredLoader
# 2. 修复：官方工具，过滤复杂元数据（解决Chroma报错）
from langchain_community.vectorstores.utils import filter_complex_metadata

# ---------------------- 核心工具函数 ----------------------
def load_and_split(file_path, text_splitter):
    """加载并分块文件（稳定无报错）"""
    try:
        # 新版加载器 + 中文支持，消除所有警告
        loader = UnstructuredLoader(
            file_path,
            extract_images=True,
            extract_tables=True,
            chunking_strategy="by_title"
        )
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        
        # 过滤复杂元数据（解决Chroma报错）
        split_docs = filter_complex_metadata(split_docs)
        
        # 绑定文件源，用于增量更新
        for doc in split_docs:
            doc.metadata["source"] = file_path
        return split_docs
    
    # 捕获文件占用/权限错误，程序不崩溃
    except PermissionError:
        print(f"❌ 错误：文件被占用，请关闭 {file_path} 后重试！")
        return []
    except Exception as e:
        print(f"❌ 读取文件失败：{str(e)}")
        return []

# ---------------------- 监听核心类 ----------------------
class DocumentUpdateHandler(FileSystemEventHandler):
    def __init__(self, record_manager, vector_store, text_splitter):
        self.record_manager = record_manager
        self.vector_store = vector_store
        self.text_splitter = text_splitter

    def on_modified(self, event):
        """文件修改自动更新"""
        if not event.is_directory and event.src_path.endswith((".pdf", ".docx", ".txt", ".md")):
            # 防抖：避免修改触发两次事件
            time.sleep(0.5)
            print(f"\n📝 检测到文件更新：{event.src_path}")
            updated_docs = load_and_split(event.src_path, self.text_splitter)
            if updated_docs:
                index(
                    updated_docs,
                    self.record_manager,
                    self.vector_store,
                    cleanup="incremental",
                    source_id_key="source"
                )

    def on_created(self, event):
        """新文件自动添加"""
        if not event.is_directory and event.src_path.endswith((".pdf", ".docx", ".txt", ".md")):
            print(f"\n🆕 检测到新文件：{event.src_path}")
            new_docs = load_and_split(event.src_path, self.text_splitter)
            if new_docs:
                index(
                    new_docs,
                    self.record_manager,
                    self.vector_store,
                    cleanup="incremental",
                    source_id_key="source"
                )

# ---------------------- 启动监听 ----------------------
def start_file_monitor(record_manager, vector_store, text_splitter, watch_path="docs/"):
    event_handler = DocumentUpdateHandler(record_manager, vector_store, text_splitter)
    observer = Observer()
    observer.schedule(event_handler, path=watch_path, recursive=False)
    observer.daemon = True
    observer.start()
    print(f"✅ 后台监听已启动：监控目录 {watch_path}")
    return observer