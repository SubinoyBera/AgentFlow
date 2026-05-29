
class NewSessionResponse(BaseModel):
    thread_id: str

class ThreadItem(BaseModel):
    thread_id: str
    title: str
    created_at: str
    last_message_at: str
    last_preview: str

class SidebarResponse(BaseModel):
    threads: list[ThreadItem]

class MessageItem(BaseModel):
    role: str
    content: str

class SessionMessagesResponse(BaseModel):
    thread_id: str
    messages: list[MessageItem]

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    thread_id: str
    reply: str

class UploadResponse(BaseModel):
    thread_id: str
    status: str
    detail: Optional[str] = None
    topic: Optional[str] = None
    summary: Optional[str] = None