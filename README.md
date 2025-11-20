# 小说问答助手 (Novel Q&A Assistant)

一个基于RAG（检索增强生成）技术的小说内容问答系统，能够智能检索小说片段并回答相关问题。

## 项目概述

本项目使用LangChain框架构建，结合DeepSeek LLM和BGE中文嵌入模型，实现了对小说内容的智能检索和问答功能。系统能够理解用户关于小说情节、人物、场景的问题，并返回准确的原文片段和解释。

## 功能特性

- 📚 **智能文本切分**: 自动将小说按章节和语义段落切分
- 🔍 **向量检索**: 使用FAISS向量数据库进行高效相似度检索
- 🤖 **智能问答**: 基于DeepSeek LLM的问答代理系统
- 📊 **相关性判断**: 自动判断检索结果与用户问题的相关性
- 🔗 **上下文获取**: 支持通过chunk_index获取相邻原文内容

## 项目结构

```
小说助手/
├── divide.py          # 文本切分模块
├── rag.py             # 向量索引构建模块
├── agent.py           # 问答代理系统
├── longzu13_index/    # FAISS向量索引目录
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包：见 `requirements.txt`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用步骤

1. **准备小说文本**
   - 将小说文本文件命名为 `longzu.txt` 放在项目根目录

2. **文本切分**
   ```bash
   python divide.py
   ```
   - 自动按章节和语义段落切分文本
   - 生成 `longzu13.jsonl` 文件

3. **构建向量索引**
   ```bash
   python rag.py
   ```
   - 使用BGE中文模型生成向量嵌入
   - 构建FAISS向量索引

4. **启动问答系统**
   ```bash
   python agent.py
   ```
   - 启动基于DeepSeek的问答代理
   - 支持自然语言问题查询

### 示例使用

```python
from agent import ask_novel_question

# 提问关于小说内容的问题
question = "路明非在电影院被赵孟华戏耍，诺诺救场这个片段的具体细节？"
answer = ask_novel_question(question)
print(answer)
```

## 技术架构

### 核心组件

1. **文本处理层** (`divide.py`)
   - 按章节切分
   - 语义段落分割
   - 重叠处理

2. **向量化层** (`rag.py`)
   - BAAI/bge-large-zh 中文嵌入模型
   - FAISS 向量数据库
   - 归一化处理

3. **问答代理层** (`agent.py`)
   - DeepSeek Chat LLM
   - LangChain Agent
   - 工具调用系统

### 工具系统

- `NovelSearch`: 关键词检索和相关性判断
- `FetchByMeta`: 根据chunk_index获取原文

## 配置说明

### API配置

项目使用DeepSeek API，需要在环境变量中配置：

```python
os.environ["OPENAI_API_KEY"] = "your-deepseek-api-key"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
```

### 模型配置

- 嵌入模型: `BAAI/bge-large-zh`
- LLM模型: `deepseek-chat`

## 性能特点

- **高效检索**: 支持大规模小说文本的快速检索
- **语义理解**: 基于深度学习的语义相似度计算
- **上下文感知**: 能够理解时间顺序和上下文关系
- **智能摘要**: 自动生成相关片段的摘要

## 注意事项

1. 确保小说文本文件编码为UTF-8
2. 首次运行需要下载BGE模型，请确保网络连接
3. 向量索引构建需要一定时间，取决于文本大小
4. 建议在GPU环境下运行以获得更好的性能

## 许可证

本项目仅供学习和研究使用。