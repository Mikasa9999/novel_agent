import os
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# =====================================
# 1️⃣ 参数设置
# =====================================
EMBED_MODEL = "BAAI/bge-large-zh"
CHUNKS_PATH = "longzu13.jsonl"  # 之前生成的 JSONL
INDEX_DIR = "longzu13_index"  # 保存索引的目录

# =====================================
# 2️⃣ 初始化向量模型（✅ 启用归一化）
# =====================================
print("🚀 正在加载 Embedding 模型:", EMBED_MODEL)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}  # ✅ 启用L2归一化 => 内积≈余弦相似度
)

# =====================================
# 3️⃣ 加载切分后的数据
# =====================================
print("📖 读取 JSONL 文件:", CHUNKS_PATH)
docs = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if not item.get("text"):
            continue

        metadata = {
            "chunk_index": int(item.get("chunk_index", -1)),
            "chapter_index": int(item.get("chapter_index", -1)),   # ✅ 改成数字索引
            "offset_in_chapter": int(item.get("offset_in_chapter", -1)),
        }
        docs.append(Document(page_content=item["text"], metadata=metadata))

print(f"✅ 加载完成，共 {len(docs)} 个文本块")

# =====================================
# 4️⃣ 创建 FAISS 向量索引（使用内积相似度）
# =====================================
print("🔍 正在构建 FAISS 向量索引（已归一化）...")
vectorstore = FAISS.from_documents(docs, embeddings)

# =====================================
# 5️⃣ 保存索引
# =====================================
vectorstore.save_local(INDEX_DIR)
print(f"💾 索引已保存到: {INDEX_DIR}")

# =====================================
# 6️⃣ 验证加载
# =====================================
print("🔎 测试加载索引并检索示例...")
loaded = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = loaded.as_retriever(search_kwargs={"k": 3})

query = "绘梨衣之死"
results = retriever.get_relevant_documents(query)
for i, r in enumerate(results):
    print(f"\n--- Top {i+1} ---")
    print(f"章节索引: {r.metadata.get('chapter_index')}")
    print(f"章节内偏移: {r.metadata.get('offset_in_chapter')}")
    print(f"全局chunk索引: {r.metadata.get('chunk_index')}")
    print(f"内容预览: {r.page_content[:]}")
