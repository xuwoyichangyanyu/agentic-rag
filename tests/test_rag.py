import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import Agent, LocalStore, split_text


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "store.json"
        self.data_patch = patch("rag.DATA_DIR", self.path.parent)
        self.store_patch = patch("rag.STORE_PATH", self.path)
        self.data_patch.start()
        self.store_patch.start()
        self.store = LocalStore()
        self.store.add_document("notes.md", "本地 RAG 系统支持文档上传与语义检索。所有数据仅保存在本机。\n\nOllama 提供本地大语言模型推理能力。")
        self.agent = Agent(self.store)

    def tearDown(self):
        self.store_patch.stop()
        self.data_patch.stop()
        self.temp.cleanup()

    def test_chunking_preserves_content(self):
        chunks = split_text("第一段内容。\n\n第二段内容。")
        self.assertEqual(len(chunks), 1)
        self.assertIn("第二段", chunks[0])

    def test_search_returns_relevant_source(self):
        results = self.store.search("什么是本地语义检索？")
        self.assertTrue(results)
        self.assertEqual(results[0]["documentName"], "notes.md")

    def test_agent_has_trace_and_citations(self):
        result = self.agent.run("系统支持什么功能？", use_model=False)
        self.assertIn("answer", result)
        self.assertGreaterEqual(len(result["trace"]), 3)
        self.assertTrue(result["sources"])

    def test_empty_store_returns_guidance(self):
        self.store.delete_document(self.store.documents[0]["id"])
        result = self.agent.run("任意问题", use_model=False)
        self.assertIn("先上传", result["answer"])


if __name__ == "__main__":
    unittest.main()
