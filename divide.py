import re
import json
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 1) 读取文本 ==========
input_path = "longzu.txt"   # 改成你的小说路径
text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
text = text.replace("\r\n", "\n").replace("\r", "\n")

# ========== 2) 第一层切分：按“空行”为章节 ==========
chapters = re.split(r"\n{2,}", text.strip())
chapters = [re.sub(r"[ \t]+", " ", c).strip() for c in chapters if c.strip()]

print(f"📖 共检测到章节数: {len(chapters)}")

# ========== 3) 第二层切分函数 ==========
def chunk_by_length(s: str, min_len: int = 1200, max_len: int = 1300):
    n = len(s)
    if n <= max_len:
        return [s]
    chunks = []
    start = 0
    while start < n:
        end = min(start + max_len, n)
        sub = s[start:end]
        last_break = None
        for m in re.finditer(r"[。！？；…\n]", sub):
            last_break = m
        if last_break:
            cut_pos = start + last_break.end()
            if (cut_pos - start) < min_len and end < n:
                cut_pos = end
        else:
            cut_pos = end
        if cut_pos <= start:
            cut_pos = min(start + max_len, n)
        chunks.append(s[start:cut_pos].strip())
        start = cut_pos
    return chunks

# ========== 4) 合并过短块 ==========
def merge_small(chunks, min_merge: int = 800):
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for c in chunks[1:]:
        if len(c) < min_merge:
            merged[-1] = (merged[-1].rstrip() + "\n" + c.lstrip()).strip()
        else:
            merged.append(c)
    return merged

# ========== 5) 添加 overlap ==========
def add_overlap(chunks, overlap: int = 200):
    if not chunks:
        return chunks
    new_chunks = []
    for i, c in enumerate(chunks):
        if i == 0:
            new_chunks.append(c)
        else:
            prev = chunks[i - 1]
            prefix = prev[-overlap:] if len(prev) > overlap else prev
            chunk = (prefix + "\n" + c).strip()
            new_chunks.append(chunk)
    return new_chunks

# ========== 6) 遍历章节并切分 ==========
all_chunks = []
chunk_index = 0

for chapter_idx, chapter_text in enumerate(chapters):
    sub = chunk_by_length(chapter_text, min_len=1200, max_len=1300)
    sub = merge_small(sub, min_merge=800)
    sub = add_overlap(sub, overlap=200)

    for offset_in_chapter, c in enumerate(sub):
        all_chunks.append({
            "chunk_index": chunk_index,
            "chapter_index": chapter_idx,
            "chapter_name": f"第{chapter_idx + 1}章",
            "offset_in_chapter": offset_in_chapter,
            "text": c
        })
        chunk_index += 1

# ========== 7) 统计信息 ==========
lengths = [len(c["text"]) for c in all_chunks]
df = pd.DataFrame({
    "chunk_index": [c["chunk_index"] for c in all_chunks],
    "chapter_index": [c["chapter_index"] for c in all_chunks],
    "offset_in_chapter": [c["offset_in_chapter"] for c in all_chunks],
    "char_len": lengths,
    "preview": [c["text"][:80].replace("\n", " ") + ("..." if len(c["text"]) > 80 else "") for c in all_chunks]
})

summary = {
    "num_chunks": len(all_chunks),
    "num_chapters": len(chapters),
    "min_len": int(min(lengths)) if lengths else 0,
    "max_len": int(max(lengths)) if lengths else 0,
    "mean_len": float(stats.mean(lengths)) if lengths else 0.0,
    "median_len": float(stats.median(lengths)) if lengths else 0.0,
    "p90_len": int(df["char_len"].quantile(0.9)) if len(df) else 0,
    "p95_len": int(df["char_len"].quantile(0.95)) if len(df) else 0,
}

# ========== 8) 保存 ==========
out_jsonl = "longzu13.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print("✅ 切分 + 章节偏移 完成")
print(f"📖 共 {summary['num_chapters']} 章，{summary['num_chunks']} 段")
print(f"📏 长度范围: {summary['min_len']} - {summary['max_len']}")
print(f"📊 平均长度: {summary['mean_len']:.1f}, 中位数: {summary['median_len']:.1f}")

# ========== 9) 可视化 ==========
plt.figure(figsize=(8, 4))
plt.hist(lengths, bins=min(50, max(10, len(lengths)//2)), edgecolor="black")
plt.title("Chunk Length Distribution (characters)")
plt.xlabel("Characters per chunk")
plt.ylabel("Count")
plt.show()

# ========== 10) 预览 ==========
print(df.head(10))
