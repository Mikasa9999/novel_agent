import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage

# ====================================
# 1️⃣ DeepSeek API 配置
# ====================================
os.environ["OPENAI_API_KEY"] = "your_deepseek_api_key_here"#也可以切换其他模型
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# ====================================
# 2️⃣ 加载向量库
# ====================================
EMBED_MODEL = "BAAI/bge-large-zh"
INDEX_PATH = "longzu13_index"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}  # ✅ 启用归一化
)
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ====================================
# 3️⃣ 全局变量：原始问题
# ====================================
GLOBAL_QUESTION = ""


# ====================================
# 4️⃣ 根据 chunk_index 获取原文
# ====================================
def fetch_by_meta(text: str) -> str:
    """
    输入格式："1234" 或 1234
    根据 chunk_index 返回原文内容。
    """
    try:
        clean_text = text.strip().strip('"').strip("'")
        chunk_index = int(clean_text)
    except Exception:
        return f"❌ 输入格式错误，应为纯数字 chunk_index，例如 '1234'（收到: {text}）"

    for doc in vectorstore.docstore._dict.values():
        meta = doc.metadata
        if int(meta.get("chunk_index", -1)) == chunk_index:
            chapter_idx = meta.get("chapter_index", -1)
            offset = meta.get("offset_in_chapter", -1)
            return f"[第{chapter_idx}章 第{offset}段 | chunk_index={chunk_index}]\n{doc.page_content.strip()}"

    return f"❌ 未找到 chunk_index={chunk_index} 对应的内容"


# ====================================
# 5️⃣ 检索 + 相关性判断
# ====================================
def search_and_judge(query_text: str) -> str:
    """
    根据 llm 生成的关键词执行检索，
    使用全局原问题 (GLOBAL_QUESTION) 判断是否相关。
    若相关返回结构化结果（chunk_index、章节索引、偏移、摘要），否则返回“无效”。
    """
    global GLOBAL_QUESTION
    original_question = GLOBAL_QUESTION or "(无)"

    # Step 1️⃣ 相似度检索（取前15）
    docs_and_scores = vectorstore.similarity_search_with_score(query_text, k=15)
    # 因为 BGE 向量经过归一化，相似度越大越相似
    docs_and_scores = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)
    top_docs = docs_and_scores[:5]

    if not top_docs:
        return "无效"

    # Step 2️⃣ 整理文本与元信息
    snippets = []
    for doc, score in top_docs:
        meta = doc.metadata
        snippets.append({
            "chunk_index": int(meta.get("chunk_index")),
            "chapter_index": int(meta.get("chapter_index")),
            "offset_in_chapter": int(meta.get("offset_in_chapter")),
            "score": round(float(score), 4),
            "text": doc.page_content.strip()
        })

    context_json = json.dumps(snippets, ensure_ascii=False, indent=2)

    # Step 3️⃣ 让 LLM 判断 + 生成摘要
    llm_judge = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2,
        openai_api_base="https://api.deepseek.com",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    prompt = f"""
你是一名小说内容分析助手。
用户的原问题是：「{original_question}」
关键词检索为：「{query_text}」

以下是根据关键词检索到的小说片段（完整文本 + 元信息）。

### 任务要求：
1. 如果所有片段都无关，请输出 “无效”。
2. 如果部分片段相关，请选择最相关的 1~2 个。
3. 每个选中片段生成一个简短摘要（≤100字），说明它与用户问题的关系。
4. 输出严格为 JSON 数组，字段如下：
   - chunk_index (数字, 来自 metadata)
   - chapter_index (数字, 来自 metadata)
   - offset_in_chapter (数字, 来自 metadata)
   - summary (摘要文本)

小说片段如下 (JSON 数据)：
{context_json}

请严格输出符合 JSON 语法的结果：
"""

    resp = llm_judge.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


# ====================================
# 6️⃣ 注册工具
# ====================================
tools = [
    Tool(
        name="NovelSearch",
        func=search_and_judge,
        description=(
            "根据关键词检索小说片段并判断是否回答了用户问题。"
            "输入关键词可以是句子，详细一点，返回的是JSON结构，包含chunk_index、chapter_index、offset_in_chapter、摘要。"
        ),
    ),
    Tool(
        name="FetchByMeta",
        func=fetch_by_meta,
        description="根据 chunk_index 获取小说原文，输入格式为'1234'。",
    )
]

# ====================================
# 7️⃣ 创建 DeepSeek Agent
# ====================================
llm_agent = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.4,
    openai_api_base="https://api.deepseek.com",
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

agent = initialize_agent(
    tools=tools,
    llm=llm_agent,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
)

# ====================================
# 8️⃣ 用户接口
# ====================================
def ask_novel_question(question: str):
    global GLOBAL_QUESTION
    GLOBAL_QUESTION = question
    return agent.invoke(
        f"请根据小说内容回答：{question}。"
        f"如有需要，可调用 NovelSearch 或 FetchByMeta 工具。"
        f"NovelSearch 工具返回的结果中含有chunk_index ，chunk_index按小说顺序排列，如果找到相关的chunk_index,请自动调用 FetchByMeta 获取完整内容。"
        f"多利用chunk_index中蕴含的时间。加减chunk_index获取其相邻的原文"
    )


# ====================================
# 9️⃣ 测试
# ====================================
if __name__ == "__main__":
    q = "路明非在电影院被赵孟华戏耍，诺诺救场这个片段的具体细节？"
    print("\n🧠 用户问题：", q)
    ans = ask_novel_question(q)
    print("\n✅ 最终回答：", ans)
