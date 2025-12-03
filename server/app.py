import os
import secrets
from datetime import datetime
from typing import Optional, Annotated

from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from starlette.background import BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from .settings import settings
from .orm import get_db, Conversation, AnalyticsEvent, TopicCluster, Message
from .llm_client import chat_stream
from .ml_engine import ml_engine
# from .services import upsert_conversation, log_analytics_event # Placeholder for full services

# --- App Setup ---
app = FastAPI(
    title=f"{settings.UNIVERSITY_NAME} Chatbot",
    description="Grounded, private university chatbot with admin analytics.",
)

# --- Database Initialization on Startup ---
@app.on_event("startup")
def startup_event():
    from .orm import engine, Base
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    print("DEBUG: Database tables created/verified.")

# CORS (Restrict this to your frontend domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be ["https://your.domain"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templating and Static Files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Authentication Dependency ---
def admin_auth_dependency(request: Request):
    """Token-gated authentication for admin APIs."""
    # 1. Check Cookie
    token_cookie = request.cookies.get("admin_token")
    if token_cookie and token_cookie == settings.ADMIN_API_TOKEN.strip():
        return True

    # 2. Check Header (Fallback)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1].strip()
        if token == settings.ADMIN_API_TOKEN.strip():
            return True
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized access to admin API.",
        headers={"WWW-Authenticate": "Bearer"},
    )

# --- Minimal Service and Analytics Placeholders ---

# In a real app, these would be in services.py and analytics_engine.py
# Using minimal versions here for a runnable app.

def simple_topic_clustering(message: str) -> Optional[int]:
    """Very simple rule-based clustering (Mode A). Returns a cluster ID."""
    message = message.lower()
    if 'registrar' in message or 'add' in message or 'drop' in message:
        return 1 # 'Registration/Enrollment'
    if 'bursar' in message or 'tuition' in message or 'bill' in message:
        return 2 # 'Billing/Bursar'
    if 'wifi' in message or 'eduroam' in message or 'network' in message:
        return 3 # 'IT/Network'
    if 'event' in message or 'activity' in message or 'club' in message or 'party' in message:
        return 4 # 'Student Life/Events'
    return None # 'Other'

def upsert_conversation(db: Session, session_id: str, user_hash: Optional[str] = None) -> Conversation:
    """Finds or creates a conversation."""
    conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
    if not conv:
        conv = Conversation(session_id=session_id, user_hash=user_hash)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        # Create initial placeholder clusters if they don't exist
        for id, label in [(1, "Registration"), (2, "Billing"), (3, "IT/Wifi"), (4, "Student Life/Events")]:
            if not db.query(TopicCluster).filter(TopicCluster.id == id).first():
                db.add(TopicCluster(id=id, label=label))
        db.commit()
    return conv

def log_analytics_event(
    db: Session, 
    conversation_id: int, 
    user_message: str,
    helpful: bool = False, 
    clicked_url: Optional[str] = None
):
    """Logs the derived analytics event with ML-powered topic detection."""
    # Use ML engine for topic prediction if trained, otherwise fallback to simple clustering
    cluster_id = ml_engine.predict_topic(user_message)
    if cluster_id is None:
        cluster_id = simple_topic_clustering(user_message)
    
    domain = None
    if clicked_url:
        try:
            from urllib.parse import urlparse
            domain = urlparse(clicked_url).netloc
        except Exception:
            pass # Keep domain None if parsing fails

    event = AnalyticsEvent(
        conversation_id=conversation_id,
        cluster_id=cluster_id,
        helpful=helpful,
        clicked_domain=domain
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

# --- Public Endpoints (Student Chat) ---

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Renders the main student chat page."""
    # Use a secure, non-guessable session ID for the user's thread
    session_id = request.cookies.get("chatbot_session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        
    response = templates.TemplateResponse("chat.html", {
        "request": request,
        "university_name": settings.UNIVERSITY_NAME,
        "session_id": session_id,
        "suggested_questions": [
            "When is the deadline to drop a class?",
            "How do I pay my tuition bill?",
            "Where can I find the academic calendar?",
            "What are the library hours?",
            "How do I apply for housing?"
        ]
    })
    response.set_cookie(key="chatbot_session_id", value=session_id, httponly=True, secure=True)
    return response

@app.post("/api/chat", response_model=None) # We use the streaming endpoint for chat, this is for non-streaming fallback/initial request setup.
async def chat_endpoint_initial(
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Form()], 
    user_message: Annotated[str, Form()],
    clicked_url: Annotated[Optional[str], Form()] = None,
    helpful: Annotated[Optional[bool], Form()] = False,
):
    """Placeholder for the initial chat POST, which immediately redirects to stream."""
    
    # 1. Upsert Conversation and Log User's Event
    conversation = upsert_conversation(db, session_id)
    log_analytics_event(
        db, 
        conversation_id=conversation.id, 
        user_message=user_message,
        helpful=helpful, 
        clicked_url=clicked_url
    )
    
    # We will use the SSE stream, so this endpoint only returns a signal
    # to the HTMX client to switch to the SSE endpoint for the actual reply.
    # In a real HTMX flow, you'd trigger the stream from the form submission.
    # For now, we return a simple response indicating success.
    
    # NOTE: The actual chat reply logic moves to /api/chat/stream for SSE.
    return JSONResponse(
        content={
            "session_id": session_id,
            "user_message": user_message,
            "status": "ready_to_stream"
        }
    )

@app.get("/api/chat/stream")
async def chat_sse_stream(
    user_message: str,
    session_id: str,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
):
    """Streams the LLM response as Server-Sent Events (SSE)."""
    
    # Helper to convert async generator to SSE format
    async def event_generator():
        # Start the stream
        full_reply_content = ""
        citations_data = None
        
        try:
            async for chunk in chat_stream(user_message):
                # Check if this chunk contains citations
                if "__CITATIONS__:" in chunk:
                    # Extract citations and don't add to content
                    citations_data = chunk
                    print(f"Captured citations in app.py: {citations_data[:100]}")
                else:
                    # Regular content chunk
                    full_reply_content += chunk
                    # SSE format: data: <chunk>\n\n
                    # Handle newlines in chunk by sending multiple data lines
                    for line in chunk.split('\n'):
                        yield f"data: {line}\n"
                    yield "\n"
        except ConnectionError as e:
            error_msg = f"[ERROR] {e}"
            for line in error_msg.split('\n'):
                yield f"data: {line}\n"
            yield "\n"
        finally:
            # Send citations BEFORE [DONE] if we have them
            if citations_data:
                print(f"Sending citations to frontend: {citations_data[:100]}")
                for line in citations_data.split('\n'):
                    yield f"data: {line}\n"
                yield "\n"
            
            # Send a final 'done' event
            yield "data: [DONE]\n\n"
            
            # 2. Log Assistant's Reply (Background Task for non-blocking)
            def log_assistant_reply():
                try:
                    conversation = db.query(Conversation).filter(Conversation.session_id == session_id).first()
                    if conversation and settings.STORE_MESSAGE_TEXT:
                        # Estimate token usage (approx 4 chars per token)
                        # Input tokens (user message) + Output tokens (assistant reply)
                        input_tokens = len(user_message) // 4
                        output_tokens = len(full_reply_content) // 4
                        total_tokens = input_tokens + output_tokens
                        
                        msg = Message(
                            conversation_id=conversation.id,
                            role='assistant',
                            content=full_reply_content,
                            model='sonar-pro',
                            token_usage=total_tokens
                        )
                        db.add(msg)
                        db.commit()
                except Exception as e:
                    print(f"Error logging assistant message in background: {e}")
            
            background_tasks.add_task(log_assistant_reply)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/feedback")
async def submit_feedback(
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Form()],
    helpful: Annotated[bool, Form()]
):
    """Records user feedback (Like/Dislike) for the latest interaction."""
    # Find the conversation
    conversation = db.query(Conversation).filter(Conversation.session_id == session_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Find the latest analytics event for this conversation
    # We assume the feedback is for the most recent query/response cycle
    latest_event = db.query(AnalyticsEvent).filter(
        AnalyticsEvent.conversation_id == conversation.id
    ).order_by(AnalyticsEvent.created_at.desc()).first()
    
    if latest_event:
        latest_event.helpful = helpful
        db.commit()
        return {"status": "success", "message": "Feedback recorded"}
    
    return {"status": "ignored", "message": "No event found to rate"}

@app.post("/api/track-click")
async def track_click(
    db: Annotated[Session, Depends(get_db)],
    session_id: Annotated[str, Form()],
    url: Annotated[str, Form()]
):
    """Records a click on a citation or resource."""
    conversation = db.query(Conversation).filter(Conversation.session_id == session_id).first()
    if not conversation:
        return {"status": "error", "message": "Conversation not found"}
        
    # We can log this as a new event or update the last one. 
    # For simplicity and distinct tracking, let's create a lightweight event 
    # or just update the last one if it doesn't have a domain yet.
    # A better approach for "Top Clicked Resources" is to just log it.
    
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Create a specific event for the click? 
        # Or better, just update the 'clicked_domain' of the last interaction if empty,
        # or create a new event if we want to track multiple clicks.
        # Let's create a new event to track every click independently.
        event = AnalyticsEvent(
            conversation_id=conversation.id,
            cluster_id=None, # It's a click, not a new query topic
            helpful=None,
            clicked_domain=domain
        )
        db.add(event)
        db.commit()
        return {"status": "success", "clicked": domain}
    except Exception as e:
        print(f"Error tracking click: {e}")
        return {"status": "error", "message": str(e)}

# --- Admin Endpoints (Visual Analytics) ---

@app.get("/admin", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Renders the admin login page."""
    return templates.TemplateResponse("admin_login.html", {
        "request": request,
        "university_name": settings.UNIVERSITY_NAME
    })

@app.post("/admin/api/login")
async def admin_login(request: Request):
    """Validates admin token and sets a secure cookie."""
    try:
        data = await request.json()
        token = data.get("token")
    except:
        form = await request.form()
        token = form.get("token")
    
    print(f"Admin login attempt with token: {token[:5]}... match={token.strip() == settings.ADMIN_API_TOKEN.strip()}")
    
    if not token or token.strip() != settings.ADMIN_API_TOKEN.strip():
        return JSONResponse({"status": "error", "message": "Invalid token"}, status_code=401)
    
    response = JSONResponse({"status": "success"})
    response.set_cookie(
        key="admin_token",
        value=token.strip(),
        httponly=True,
        secure=False, # Set to True in production with HTTPS
        samesite="lax",
        path="/",
        max_age=86400 # 24 hours
    )
    return response

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Renders the admin dashboard page (requires valid token in cookie)."""
    token_cookie = request.cookies.get("admin_token")
    if not token_cookie or token_cookie != settings.ADMIN_API_TOKEN.strip():
        return RedirectResponse(url="/admin")

    return templates.TemplateResponse("admin.html", {
        "request": request,
        "university_name": settings.UNIVERSITY_NAME
        # Token is NO LONGER injected here
    })

@app.get("/admin/api/overview", dependencies=[Depends(admin_auth_dependency)])
async def get_overview(
    db: Annotated[Session, Depends(get_db)],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
):
    """Returns high-level KPIs for the admin dashboard."""
    total_threads = db.query(Conversation).count()
    total_questions = db.query(AnalyticsEvent).count()
    avg_questions_per_thread = total_questions / total_threads if total_threads > 0 else 0

    # Top domains clicked
    top_domains_query = db.query(
        AnalyticsEvent.clicked_domain,
        func.count().label('count')
    ).filter(AnalyticsEvent.clicked_domain.isnot(None)).group_by(AnalyticsEvent.clicked_domain).order_by(func.count().desc()).limit(5).all()
    
    top_domains = [{"domain": d, "count": c} for d, c in top_domains_query]

    # Calculate satisfaction rate
    total_feedback = db.query(AnalyticsEvent).filter(AnalyticsEvent.helpful.isnot(None)).count()
    helpful_count = db.query(AnalyticsEvent).filter(AnalyticsEvent.helpful == True).count()
    satisfaction_rate = (helpful_count / total_feedback * 100) if total_feedback > 0 else 0

    # Calculate total tokens used
    total_tokens = db.query(func.sum(Message.token_usage)).filter(Message.token_usage.isnot(None)).scalar() or 0
    
    # Estimate cost (Perplexity pricing: ~$1 per 1M tokens for sonar-pro)
    estimated_cost = (total_tokens / 1_000_000) * 1.0

    return {
        "total_threads": total_threads,
        "total_questions": total_questions,
        "avg_questions_per_thread": round(avg_questions_per_thread, 1),
        "top_domains": top_domains,
        "satisfaction_rate": round(satisfaction_rate, 1),
        "total_tokens": total_tokens,
        "estimated_cost": round(estimated_cost, 2)
    }

# Minimal implementation for one chart endpoint
@app.get("/admin/api/top-clusters", dependencies=[Depends(admin_auth_dependency)])
async def get_top_clusters(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 5,
    start: Optional[datetime] = None, 
    end: Optional[datetime] = None
):
    """Returns data for the Top Topics/Niches horizontal bar chart."""
    # NOTE: Real query logic needed for date filtering
    
    cluster_data = db.query(
        TopicCluster.label,
        func.count(AnalyticsEvent.id)
    ).outerjoin(AnalyticsEvent).group_by(TopicCluster.label).order_by(func.count().desc()).limit(limit).all()
    
    return [
        {"cluster_id": label.lower().replace('/', '_').replace(' ', '_'), "label": label, "count": count} 
        for label, count in cluster_data
    ]

# ML-Powered Insights Endpoint
@app.get("/admin/api/ml-insights", dependencies=[Depends(admin_auth_dependency)])
async def get_ml_insights(
    db: Annotated[Session, Depends(get_db)],
    days: int = 7
):
    """Returns ML-powered insights and predictions."""
    return ml_engine.get_conversation_insights(db, days)

# Train ML Model Endpoint
@app.post("/admin/api/train-ml", dependencies=[Depends(admin_auth_dependency)])
async def train_ml_model(
    db: Annotated[Session, Depends(get_db)],
    n_clusters: int = 5
):
    """Train the ML topic clustering model on historical data."""
    # Get all messages from the database
    messages = db.query(Message).filter(Message.role == 'user').limit(1000).all()
    
    if len(messages) < 10:
        return {"status": "error", "message": "Not enough data to train. Need at least 10 messages."}
    
    message_texts = [msg.content for msg in messages]
    result = ml_engine.train_topic_clusters(message_texts, n_clusters)
    
    # Update topic clusters in database
    if result["status"] == "success":
        for cluster_info in result["clusters"]:
            cluster = db.query(TopicCluster).filter(TopicCluster.id == cluster_info["cluster_id"] + 1).first()
            if not cluster:
                cluster = TopicCluster(
                    id=cluster_info["cluster_id"] + 1,
                    label=cluster_info["label"]
                )
                db.add(cluster)
            else:
                cluster.label = cluster_info["label"]
        db.commit()
    
    return result

# --- Healthcheck ---
@app.get("/healthz")
def health_check():
    """Simple health check endpoint."""
    try:
        # Check DB connection
        db = next(get_db())
        db.execute(db.text("SELECT 1"))
        
        # Check API key presence (already done by settings)
        if not settings.PERPLEXITY_API_KEY:
             return JSONResponse({"status": "error", "message": "PERPLEXITY_API_KEY missing"}, status_code=503)

        return {"status": "ok", "db": "connected", "llm_client": "configured"}
    except Exception as e:
        return JSONResponse({"status": "error", "db": str(e)}, status_code=503)