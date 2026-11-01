# Lattice — Local Agentic RAG

一个本地优先的 Agentic RAG 系统。上传 PDF、TXT 或 Markdown 文档后，Agent 会规划检索、评估相关性、在低相关性时重试，并返回带来源的回答与完整执行轨迹。

## 启动

```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

打开 `http://localhost:8000`。也可运行 `npm run dev`。

## 本地模型（可选）

安装并启动 [Ollama](https://ollama.com)，然后拉取模型：

```bash
ollama pull qwen2.5:3b
```

默认连接 `http://localhost:11434`，可在界面“模型设置”中修改。未运行 Ollama 时，系统仍可使用本地混合检索与抽取式回答，并会明确标注回答模式。

## 特性

- 本地文档解析、重叠切片、384 维哈希向量索引
- 向量相似度与词汇匹配混合召回
- 规划、检索、反思重试、生成回答的 Agent 流程
- 来源片段、相关性分数与 Agent 轨迹可视化
- 文档上传、删除与本地 JSON 持久化
- 不依赖云端 embedding 或云端 LLM API

## 测试

```bash
python -m unittest discover -s tests -v
```

文档和索引保存在运行目录的 `data/store.json`，该目录已被 Git 忽略。
