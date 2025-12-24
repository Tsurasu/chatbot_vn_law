# chatbot_vn_law
Hệ thống cho phép user đặt câu hỏi về pháp luật và nhận câu trả lời dựa trên knowledge base có sẵn, với khả năng stream response real-time và hiển thị sources để user kiểm tra.
# NOTE - Hệ Thống VN Law RAG

## 1. Tech Stack (Công Nghệ Sử Dụng)

### Backend
- **FastAPI**: Framework để xây dựng REST API, xử lý các request từ frontend
- **LangChain**: Thư viện để kết nối với LLM (Ollama), quản lý prompts và chuỗi xử lý
- **LangChain Ollama**: Wrapper để tích hợp Ollama với LangChain
- **Uvicorn**: Web server để chạy FastAPI app
- **Pydantic**: Validate và serialize data

### Frontend
- **React**: Framework JavaScript để xây dựng giao diện người dùng
- **Vite**: Build tool nhanh, hỗ trợ development và build production
- **React Markdown**: Render markdown cho câu trả lời

### Machine Learning & AI
- **Ollama**: Local LLM server, chạy model `qwen3:32b` để generate câu trả lời
- **Sentence Transformers**: Tạo embeddings từ text (sử dụng model local `vn-law-embedding`)
- **PyTorch**: Backend cho sentence transformers

### Vector Database & Search
- **FAISS (Facebook AI Similarity Search)**: Vector database để lưu trữ và tìm kiếm embeddings
- **BM25**: Thuật toán keyword-based search để tìm kiếm chính xác
- **Hybrid Search**: Kết hợp vector search (semantic) và BM25 (keyword) để tìm kiếm tốt hơn

### Data Processing
- **Hugging Face Datasets**: Load và xử lý datasets từ local storage
- **NumPy**: Xử lý arrays và embeddings
- **Pandas**: Xử lý data nếu cần

### Utilities
- **python-dotenv**: Load environment variables từ file .env
- **httpx**: HTTP client để giao tiếp với Ollama API
- **tqdm**: Progress bar khi xử lý data


## 2. Quy Trình Xây Dựng Hệ Thống (Build Index)

Đây là quy trình để chuẩn bị dữ liệu và xây dựng vector database từ datasets pháp luật Việt Nam.

### Bước 1: Load Datasets
- Load datasets từ thư mục local `dataset/`
- Có 2 datasets:
  - `vn-law-corpus`: Chứa các văn bản pháp luật
  - `vn-law-questions-and-corpus`: Chứa câu hỏi và câu trả lời
- Script: `scripts/build_index.py` sử dụng `DatasetLoader` từ `src/data_loader.py`

### Bước 2: Xử Lý Documents
- **Chunking**: Chia nhỏ các documents dài thành các chunks nhỏ hơn (khoảng 512 tokens mỗi chunk)
- **Cleaning**: Làm sạch text (remove extra spaces, normalize...)
- Mỗi chunk có metadata: `id`, `text`, `source`, `chunk_index`, `original_id`

### Bước 3: Tạo Embeddings
- Sử dụng model embedding local `vn-law-embedding` (được train trên data pháp luật Việt Nam)
- Model này chuyển text thành vector số (embeddings) để so sánh ngữ nghĩa
- Xử lý theo batch để tối ưu tốc độ (mặc định batch_size = 32)

### Bước 4: Xây Dựng FAISS Index
- Lưu tất cả embeddings vào FAISS index
- FAISS cho phép tìm kiếm nhanh các vectors tương tự nhau
- Có 2 loại index:
  - **Flat**: Tìm kiếm chính xác (chậm hơn nhưng chính xác hơn)
  - **IVF**: Tìm kiếm gần đúng (nhanh hơn cho datasets lớn)

### Bước 5: Xây Dựng BM25 Index
- Xây dựng BM25 index từ tất cả documents để hỗ trợ keyword search
- BM25 tốt cho tìm kiếm từ khóa chính xác

### Bước 6: Lưu Vector Store
- Lưu FAISS index, metadata và config vào thư mục `vector_db/`
- Files được lưu:
  - `index.faiss`: FAISS index
  - `metadata.pkl`: Metadata của các documents
  - `config.pkl`: Cấu hình của index

### Chạy Build Index:
```bash
python scripts/build_index.py --index-type flat --batch-size 32 --output-dir ./vector_db
```


## 3. Luồng Xử Lý Câu Hỏi (RAG Flow)

Đây là quy trình từ khi user đặt câu hỏi đến khi nhận được câu trả lời.

```
User → Frontend → Backend API → Retrieval → RAG Chain → Ollama LLM → Response → Frontend → User
```

### Chi Tiết Từng Bước:

#### Bước 1: User Nhập Câu Hỏi
- User nhập câu hỏi về pháp luật Việt Nam vào giao diện React
- Frontend (`frontend/src/components/Chat.jsx`) lấy câu hỏi

#### Bước 2: Frontend Gửi Request
- Frontend gọi API endpoint `/api/chat/stream` với:
  - `question`: Câu hỏi của user
  - `conversation_history`: Lịch sử 2 cặp câu hỏi-trả lời gần nhất (nếu có)
  - `top_k`: Số documents muốn retrieve (mặc định 10)
  - `temperature`: Độ sáng tạo của LLM (0.0-1.0, mặc định 0.7)
- Sử dụng Server-Sent Events (SSE) để stream response

#### Bước 3: Backend Nhận Request
- `api/routes/chat.py` nhận request qua endpoint `/api/chat/stream`
- Lấy RAG chain instance từ `api/dependencies.py` (singleton pattern)

#### Bước 4: Retrieval - Tìm Kiếm Documents
- `build/rag_chain.py` → `src/retrieval.py` → `RetrievalSystem.retrieve()`
- **Vector Search**:
  - Chuyển câu hỏi thành embedding bằng `VNLawEmbedder`
  - Tìm kiếm trong FAISS index để lấy các documents có embedding gần nhất (cosine similarity)
  - Lấy top 30 candidates (3x top_k) để re-rank
- **BM25 Search**:
  - Tokenize câu hỏi thành các từ
  - Tính BM25 scores cho tất cả documents
  - Normalize scores về 0-1
- **Hybrid Search**:
  - Kết hợp vector score và BM25 score với weight (mặc định: 40% vector, 60% BM25)
  - Hybrid score = (1 - weight) × vector_score + weight × BM25_score
  - Sort lại và lấy top_k documents
- Kết quả: Danh sách documents liên quan với scores

#### Bước 5: Format Prompt
- `build/prompts.py` → `PromptTemplates.format_rag_prompt()`
- Tạo prompt với cấu trúc:
  ```
  System Prompt (hướng dẫn LLM cách trả lời)
  
  ## Lịch sử cuộc trò chuyện (nếu có)
  [2 cặp Q&A gần nhất]
  
  ## Context (Các văn bản pháp luật liên quan)
  [Danh sách documents đã retrieve, mỗi doc có id, source, text]
  
  ## Câu hỏi hiện tại
  [Câu hỏi của user]
  
  ## Câu trả lời:
  ```
- System prompt hướng dẫn LLM:
  - Trả lời dựa trên context và lịch sử
  - Không bịa đặt thông tin cụ thể (số điện thoại, địa chỉ, số điều luật...)
  - Trả lời bằng tiếng Việt, chuyên nghiệp

#### Bước 6: Generate Answer với Ollama
- `build/ollama_client.py` → `OllamaClient.stream()`
- Kết nối đến Ollama server (mặc định `http://localhost:11434`)
- Gửi prompt đến model `qwen3:32b`
- Nhận response dạng stream (từng chunk một)

#### Bước 7: Stream Response về Frontend
- Backend stream các chunks về frontend qua SSE
- Mỗi chunk có format:
  ```json
  {
    "type": "answer" | "sources" | "error",
    "content": "...",
    "done": false
  }
  ```
- Frontend cập nhật UI real-time khi nhận từng chunk

#### Bước 8: Hiển Thị Kết Quả
- Frontend hiển thị:
  - **Answer**: Câu trả lời được format markdown
  - **Sources**: Danh sách các documents tham khảo với scores
- User có thể xem sources để kiểm tra độ tin cậy


## 4. Kiến Trúc Hệ Thống

### Tổng Quan
Hệ thống có kiến trúc 3 tầng: **Frontend** - **Backend API** - **RAG System**

### Frontend Layer
**Location**: `frontend/`

**Components**:
- `App.jsx`: Component chính, quản lý state và health check
- `Chat.jsx`: Component chat, xử lý input và hiển thị messages
- `Message.jsx`: Component hiển thị từng message (user hoặc assistant)
- `Loading.jsx`: Component loading animation
- `api.js`: Service để gọi backend API

**Chức năng**:
- UI/UX cho user tương tác
- Gửi câu hỏi và nhận response streaming
- Hiển thị câu trả lời và sources
- Quản lý conversation history

### Backend API Layer
**Location**: `api/`

**Files chính**:
- `main.py`: FastAPI app, khởi tạo và cấu hình routes, CORS
- `routes/chat.py`: Endpoints `/api/chat` và `/api/chat/stream`
- `routes/health.py`: Endpoint health check
- `dependencies.py`: Dependency injection, tạo RAG chain singleton
- `models/schemas.py`: Pydantic models để validate request/response

**Chức năng**:
- Nhận HTTP requests từ frontend
- Route requests đến RAG system
- Stream responses về frontend
- Handle errors và validation

### RAG System Layer
**Location**: `build/` và `src/`

**Components chính**:

#### 4.1. RAG Chain (`build/rag_chain.py`)
- **VNLawRAGChain**: Class chính điều phối toàn bộ flow
- **Chức năng**:
  - Gọi retrieval system để tìm documents
  - Format prompt với context và conversation history
  - Gọi Ollama client để generate answer
  - Trả về answer + sources + metadata
  - Hỗ trợ cả invoke (sync) và stream (async)

#### 4.2. Retrieval System (`src/retrieval.py`)
- **RetrievalSystem**: Class quản lý việc tìm kiếm documents
- **Components**:
  - **VNLawEmbedder** (`src/embedding.py`): Tạo embeddings từ text
  - **FAISSVectorStore** (`src/vector_store.py`): Lưu trữ và tìm kiếm vectors
  - **BM25 Index**: Keyword-based search
- **Chức năng**:
  - Hybrid search: Kết hợp vector search và BM25
  - Re-ranking: Sắp xếp lại kết quả bằng hybrid scores
  - Trả về top_k documents với scores

#### 4.3. Ollama Client (`build/ollama_client.py`)
- **OllamaClient**: Wrapper để giao tiếp với Ollama server
- **Chức năng**:
  - Kết nối đến Ollama API
  - Parse prompts thành messages (system + user)
  - Generate text (sync và stream)
  - Health check Ollama server

#### 4.4. Prompt Templates (`build/prompts.py`)
- **PromptTemplates**: Class quản lý việc format prompts
- **Chức năng**:
  - Format RAG prompt với context và conversation history
  - Giới hạn độ dài context để tiết kiệm memory
  - Format sources info để hiển thị

#### 4.5. Config (`build/config.py`)
- **RAGConfig**: Class chứa tất cả cấu hình
- **Cấu hình**:
  - Ollama: URL, model name, timeout
  - Vector store: Đường dẫn, embedding model
  - RAG: top_k, temperature, max_tokens
  - API: host, port, reload
  - System prompt

### Data Layer

#### Vector Database (`vector_db/`)
- FAISS index: `index.faiss`
- Metadata: `metadata.pkl`
- Config: `config.pkl`

#### Embedding Model (`embedding_model_local/`)
- Model local `vn-law-embedding` được train trên data pháp luật Việt Nam

#### Datasets (`dataset/`)
- `vn-law-corpus/`: Văn bản pháp luật
- `vn-law-questions-and-corpus/`: Câu hỏi và câu trả lời

### Luồng Dữ Liệu Tổng Quan

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────┐      HTTP/SSE       ┌─────────────┐
│  Frontend   │ ◄────────────────── │ Backend API │
│   React     │                     │   FastAPI   │
└─────────────┘                     └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  RAG Chain  │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │  Retrieval   │      │   Prompt     │      │   Ollama     │
            │   System     │      │  Templates   │      │   Client     │
            └──────┬───────┘      └──────────────┘      └──────┬───────┘
                   │                                           │
                   ├─── Embedder ───┐                          │
                   │                 │                         │
                   ├─── FAISS ───────┼─────────────────────────┤
                   │                 │                         │
                   └─── BM25 ────────┘                         │
                                                               │
                                                               ▼
                                                        ┌──────────────┐
                                                        │ Ollama Server│
                                                        │  (qwen3:32b) │
                                                        └──────────────┘
```

### Điểm Đặc Biệt

1. **Singleton Pattern**: RAG chain được tạo một lần và tái sử dụng (trong `dependencies.py`)
2. **Streaming**: Hỗ trợ stream response để UX tốt hơn
3. **Conversation History**: Lưu 2 cặp Q&A gần nhất để context tốt hơn
4. **Hybrid Search**: Kết hợp semantic (vector) và keyword (BM25) search
5. **Local Everything**: Tất cả models và datasets đều local, không cần internet khi chạy


## 5. Cách Chạy Hệ Thống

### Setup Backend
1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Setup environment variables (tạo file `.env`):
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:32b
VECTOR_STORE_DIR=./vector_db
EMBEDDING_MODEL_NAME=embedding_model_local/vn-law-embedding
```

3. Đảm bảo Ollama đang chạy với model `qwen3:32b`

4. Chạy backend:
```bash
python -m api.main
# hoặc
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Setup Frontend
1. Vào thư mục frontend:
```bash
cd frontend
```

2. Cài đặt dependencies:
```bash
npm install
```

3. Chạy frontend:
```bash
npm run dev
```

4. Mở browser: `http://localhost:5173` (port mặc định của Vite)

### Build Index (Lần Đầu)
Nếu chưa có vector database, cần build index trước:
```bash
python scripts/build_index.py --index-type flat --output-dir ./vector_db
```


## 6. Kết Luận

Hệ thống VN Law RAG là một hệ thống hỏi đáp về pháp luật Việt Nam sử dụng:
- **RAG (Retrieval Augmented Generation)**: Kết hợp tìm kiếm và generation
- **Hybrid Search**: Vector + BM25 để tìm kiếm tốt nhất
- **Local LLM**: Sử dụng Ollama với model qwen3:32b
- **Modern Stack**: FastAPI + React để tạo web app hoàn chỉnh
# VN Law RAG Frontend

Frontend React đơn giản để tương tác với VN Law RAG API.

## Yêu cầu

- Node.js 16+ và npm/yarn
- API server đang chạy tại `http://localhost:8000`

## Cài đặt

### 1. Cài đặt dependencies

```bash
cd frontend
npm install
```

### 2. Cấu hình (tùy chọn)

Tạo file `.env` trong thư mục `frontend/` nếu API chạy ở URL khác:

```env
VITE_API_URL=http://localhost:8000
```

## Sử dụng

### Development mode

```bash
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

### Build cho production

```bash
npm run build
```

Files build sẽ được tạo trong thư mục `dist/`

### Preview production build

```bash
npm run preview
```

## Tính năng

- ✅ Chat interface đơn giản
- ✅ Streaming response (real-time)
- ✅ Hiển thị sources (nguồn tham khảo)
- ✅ Health check status
- ✅ Responsive design

## Cấu trúc

```
frontend/
├── src/
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # React entry point
│   ├── components/
│   │   ├── Chat.jsx         # Chat interface
│   │   ├── Message.jsx      # Message display
│   │   └── Loading.jsx       # Loading indicator
│   ├── services/
│   │   └── api.js           # API service functions
│   └── styles/
│       └── App.css          # Main styles
├── index.html
├── package.json
└── vite.config.js
```

## API Integration

Frontend gọi các endpoints sau:

- `POST /api/chat` - Chat không streaming
- `POST /api/chat/stream` - Chat với streaming (SSE)
- `GET /api/health` - Health check

## Troubleshooting

### Không kết nối được với API

1. Đảm bảo API server đang chạy tại `http://localhost:8000`
2. Kiểm tra CORS settings trong FastAPI
3. Kiểm tra file `.env` nếu đã thay đổi API URL

### Lỗi khi build

```bash
# Xóa node_modules và cài lại
rm -rf node_modules package-lock.json
npm install
```

## License

Apache 2.0
