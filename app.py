from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

from rag import Agent, LocalStore

ROOT = Path(__file__).parent
app = FastAPI(title="Local Agentic RAG", version="1.0.0")
store = LocalStore()
agent = Agent(store)


class Query(BaseModel):
    question: str
    topK: int = 5
    useModel: bool = True


class Settings(BaseModel):
    model: str
    ollamaUrl: str


@app.get("/api/health")
def health():
    return {"status": "ok", "documents": len(store.documents), "chunks": len(store.chunks), **store.settings}


@app.get("/api/documents")
def documents():
    return sorted(store.documents, key=lambda doc: doc["createdAt"], reverse=True)


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, "仅支持 TXT、Markdown 和 PDF 文件")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(400, "文件不能超过 20 MB")
    try:
        if suffix == ".pdf":
            import io
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
        else:
            text = content.decode("utf-8-sig")
    except Exception as error:
        raise HTTPException(400, f"无法解析文件：{error}") from error
    try:
        return store.add_document(Path(file.filename).name, text)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    store.delete_document(doc_id)
    return {"ok": True}


@app.post("/api/query")
def query(payload: Query):
    if not payload.question.strip():
        raise HTTPException(400, "问题不能为空")
    return agent.run(payload.question.strip(), max(1, min(payload.topK, 10)), payload.useModel)


@app.get("/api/settings")
def get_settings():
    return store.settings


@app.put("/api/settings")
def update_settings(settings: Settings):
    store.settings.update(settings.model_dump())
    store.save()
    return store.settings


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
