"""Dependency-light, local-first agentic retrieval augmented generation."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
STORE_PATH = DATA_DIR / "store.json"
DIMENSIONS = 384


def tokens(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    bigrams = [part[i:i + 2] for part in chinese for i in range(len(part) - 1)]
    return words + bigrams


def embed(text: str) -> np.ndarray:
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % DIMENSIONS
        sign = 1 if digest[4] % 2 else -1
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def split_text(text: str, size: int = 650, overlap: int = 100) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        while len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size])
            paragraph = paragraph[size - overlap:]
        if len(current) + len(paragraph) + 1 > size and current:
            chunks.append(current)
            current = current[-overlap:]
        current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


class LocalStore:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.documents = []
        self.chunks = []
        self.settings = {"model": "qwen2.5:3b", "ollamaUrl": "http://localhost:11434"}
        self.load()

    def load(self):
        if STORE_PATH.exists():
            data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            self.documents = data.get("documents", [])
            self.chunks = data.get("chunks", [])
            self.settings.update(data.get("settings", {}))

    def save(self):
        STORE_PATH.write_text(json.dumps({"documents": self.documents, "chunks": self.chunks, "settings": self.settings}, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(self, name: str, text: str) -> dict:
        clean = text.strip()
        if not clean:
            raise ValueError("文档中没有可读取的文字")
        pieces = split_text(clean)
        doc_id = hashlib.sha256(f"{name}:{time.time_ns()}".encode()).hexdigest()[:12]
        document = {"id": doc_id, "name": name, "characters": len(clean), "chunks": len(pieces), "createdAt": int(time.time())}
        with self.lock:
            self.documents.append(document)
            for index, piece in enumerate(pieces):
                self.chunks.append({"id": f"{doc_id}-{index}", "documentId": doc_id, "documentName": name, "index": index, "text": piece, "vector": embed(piece).tolist()})
            self.save()
        return document

    def delete_document(self, doc_id: str):
        with self.lock:
            self.documents = [doc for doc in self.documents if doc["id"] != doc_id]
            self.chunks = [chunk for chunk in self.chunks if chunk["documentId"] != doc_id]
            self.save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks:
            return []
        query_vector = embed(query)
        query_tokens = set(tokens(query))
        scored = []
        for chunk in self.chunks:
            vector = np.asarray(chunk["vector"], dtype=np.float32)
            semantic = float(np.dot(query_vector, vector))
            chunk_tokens = set(tokens(chunk["text"]))
            lexical = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            score = semantic * 0.65 + lexical * 0.35
            scored.append({**{k: v for k, v in chunk.items() if k != "vector"}, "score": round(score, 4)})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


class Agent:
    def __init__(self, store: LocalStore):
        self.store = store

    def run(self, question: str, top_k: int = 5, use_model: bool = True) -> dict:
        trace = [{"step": 1, "type": "plan", "title": "理解问题", "detail": f"分析问题并生成检索计划：{question}"}]
        results = self.store.search(question, top_k)
        trace.append({"step": 2, "type": "retrieve", "title": "检索知识库", "detail": f"召回 {len(results)} 个相关片段", "count": len(results)})
        if not results:
            return {"answer": "知识库中还没有文档。请先上传 TXT、Markdown 或 PDF 文件，再开始提问。", "sources": [], "trace": trace, "mode": "retrieval"}
        best = results[0]["score"]
        refined = False
        if best < 0.12:
            keywords = " ".join(tokens(question)[:18])
            second = self.store.search(keywords, top_k)
            known = {item["id"] for item in results}
            results.extend(item for item in second if item["id"] not in known)
            results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
            refined = True
            trace.append({"step": 3, "type": "reflect", "title": "反思并重试", "detail": "首轮相关性较低，已提取关键词并扩大检索"})
        sources = results[:top_k]
        context = "\n\n".join(f"[{i + 1}] {item['text']}" for i, item in enumerate(sources))
        answer, mode = self.generate(question, context, use_model)
        trace.append({"step": len(trace) + 1, "type": "answer", "title": "生成回答", "detail": f"使用 {'Ollama 本地模型' if mode == 'ollama' else '本地抽取式回答'}，引用 {len(sources)} 个来源"})
        return {"answer": answer, "sources": sources, "trace": trace, "mode": mode, "refined": refined}

    def generate(self, question: str, context: str, use_model: bool) -> tuple[str, str]:
        if use_model:
            try:
                prompt = f"你是严谨的本地知识库助手。只能依据给定资料回答；资料不足时明确说明。使用中文，并在相关句末使用[1][2]形式标注来源。\n\n资料：\n{context}\n\n问题：{question}"
                payload = json.dumps({"model": self.store.settings["model"], "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}).encode()
                request = urllib.request.Request(self.store.settings["ollamaUrl"].rstrip("/") + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    answer = json.loads(response.read()).get("response", "").strip()
                if answer:
                    return answer, "ollama"
            except Exception:
                pass
        excerpts = []
        for i, chunk in enumerate(context.split("\n\n")[:3]):
            text = re.sub(r"^\[\d+\]\s*", "", chunk)
            excerpts.append(f"{text[:420]}{'…' if len(text) > 420 else ''} [{i + 1}]")
        return "以下为知识库中最相关的内容（未检测到可用的 Ollama 模型，当前为抽取式回答）：\n\n" + "\n\n".join(excerpts), "retrieval"
