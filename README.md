
# Finance RAG

> An agentic financial research system that combines multimodal RAG, live financial data, web search, and self-critique using LangGraph.

Finance RAG is an AI-powered financial research workspace that allows users to upload financial documents and ask questions about them while also giving the agent access to live financial data and web search.

The system combines:

- Multimodal document ingestion
- Retrieval-Augmented Generation (RAG)
- Stateful ReAct agents
- LangGraph orchestration
- Financial Modeling Prep (FMP)
- Tavily web search
- Self-critique and refinement
- RAGAs-based evaluation

---

## Features

### 1. Multimodal Document Ingestion

Supports:

- PDF
- DOCX
- Images

The ingestion pipeline automatically detects the input type and extracts the relevant information.

For PDFs:

- Extracts selectable text using `pdfminer.six`
- Detects scanned/flat PDFs
- Falls back to Tesseract OCR when necessary

For DOCX:

- Extracts paragraphs
- Extracts table contents
- Preserves document structure where possible

For images:

- Performs OCR using Tesseract

The extracted content is then:

```text
Document
   ↓
File-Type Detection
   ↓
Text / OCR Extraction
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
ChromaDB
````

---

### 2. Retrieval-Augmented Generation

Extracted documents are divided into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

Default configuration:

```text
Chunk size: 500 characters
Overlap:     50 characters
```

Each chunk is converted into an embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are persisted locally in ChromaDB.

Each chunk contains metadata such as:

* Source document
* Document type
* Chunk identifier
* Original source information

This allows the system to retrieve the most relevant information when answering a question.

---

### 3. Stateful ReAct Agent

The reasoning engine is implemented using **LangGraph**.

Instead of following a fixed linear pipeline, the agent maintains state and dynamically decides which tools are required.

The agent has access to:

| Tool                         | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| `search_financial_documents` | Search uploaded financial documents              |
| `get_stock_quote`            | Retrieve current stock information               |
| `get_company_profile`        | Retrieve company information                     |
| `get_financial_statements`   | Retrieve financial fundamentals                  |
| `search_financial_news`      | Search the web for current financial information |

For example:

```text
User:
"Compare the revenue in my uploaded report with Apple's latest financial performance."
```

The agent can determine that it needs:

```text
Document Retrieval
       +
FMP Financial Statements
       ↓
Context Combination
       ↓
Answer Generation
```

The agent does not need to call every available tool for every query.

---

### 4. Live Financial Data

Finance RAG integrates with **Financial Modeling Prep (FMP)** to retrieve financial information.

Implemented tools include:

#### Stock Quote

Retrieves information such as:

* Current price
* Trading volume
* Market capitalization

#### Company Profile

Retrieves:

* Company name
* Industry
* Sector
* Description
* Employee information

#### Financial Statements

Retrieves fundamental financial information such as:

* Revenue
* Net income
* Gross profit
* Operating income
* Financial statement data

---

### 5. Web Search

Finance RAG integrates **Tavily** to provide current web information.

The web search tool is useful for questions involving:

* Recent financial news
* Company developments
* Analyst commentary
* Current events

This allows the system to combine information from uploaded documents with information that may not exist inside those documents.

---

### 6. Self-Critique & Refinement

The generated answer is not immediately returned to the user.

The system uses a self-critique loop implemented through LangGraph.

```text
Retrieved Context
       ↓
Draft Answer
       ↓
Critique
       ↓
Quality Score
       ↓
   ┌───────────────┐
   │ Score >= 0.85 │
   └───────┬───────┘
           │
        YES│
           ↓
      Final Answer

        NO
        ↓
     Rewrite
        ↓
     Critique
        ↓
     Re-evaluate
```

Configuration:

```text
QUALITY_THRESHOLD = 0.85
MAX_ITERATIONS = 3
```

The critique evaluates:

* Faithfulness to retrieved information
* Relevance to the question
* Completeness
* Factual consistency
* Unsupported claims

If the answer does not satisfy the quality threshold, the system attempts to refine it.

---

# Architecture

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   LangGraph Agent   │
                         │   Stateful ReAct    │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             ┌───────────┐   ┌────────────┐   ┌────────────┐
             │ ChromaDB  │   │    FMP     │   │   Tavily   │
             │ Document  │   │ Financial  │   │ Web Search │
             │ Retrieval │   │    Data    │   │            │
             └─────┬─────┘   └──────┬─────┘   └──────┬─────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Context Fusion    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Draft Generation  │
                         │      Groq LLM       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Self-Critique    │
                         └──────────┬──────────┘
                                    │
                              Score < 0.85?
                              /           \
                            YES            NO
                             │              │
                             ▼              ▼
                          Rewrite       Final Answer
                             │
                             └──────→ Critique
```

---

# Document Ingestion Architecture

```text
             PDF / DOCX / IMAGE
                     │
                     ▼
              File Detection
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
        PDF         DOCX      IMAGE
          │          │          │
          ▼          ▼          ▼
      pdfminer    python-docx  Tesseract
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
               Text Cleaning
                     │
                     ▼
                Chunking
                     │
                     ▼
            HuggingFace Embeddings
                     │
                     ▼
                  ChromaDB
```

---

# Technology Stack

## AI & Orchestration

| Technology    | Purpose                           |
| ------------- | --------------------------------- |
| Python        | Core development language         |
| LangChain     | LLM, prompt and tool abstractions |
| LangGraph     | Stateful agent orchestration      |
| Groq          | LLM inference                     |
| Llama 3.3 70B | Reasoning and generation          |

---

## Retrieval & Vector Storage

| Technology       | Purpose                |
| ---------------- | ---------------------- |
| ChromaDB         | Vector database        |
| LangChain Chroma | ChromaDB integration   |
| HuggingFace      | Local embedding models |
| all-MiniLM-L6-v2 | Text embeddings        |

---

## Document Processing

| Technology   | Purpose                       |
| ------------ | ----------------------------- |
| pdfminer.six | PDF text extraction           |
| python-docx  | DOCX parsing                  |
| Tesseract    | OCR                           |
| pytesseract  | Python interface to Tesseract |

---

## External APIs

| Technology              | Purpose               |
| ----------------------- | --------------------- |
| Financial Modeling Prep | Financial market data |
| Tavily                  | Web search            |

---

## Evaluation & Testing

| Technology | Purpose           |
| ---------- | ----------------- |
| RAGAs      | RAG evaluation    |
| Pytest     | Automated testing |

---

# Project Structure

```text
finance-rag/
│
├── app/
│   ├── ingestion/
│   │   ├── detector.py
│   │   ├── ingest.py
│   │   ├── cleaner.py
│   │   └── parsers/
│   │       ├── pdf_parser.py
│   │       ├── docx_parser.py
│   │       └── image_parser.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── retriever/
│   │   └── vector_store.py
│   │
│   ├── tools/
│   │   ├── document_search.py
│   │   ├── fmp_tools.py
│   │   └── tavily_tools.py
│   │
│   ├── agent/
│   │   ├── state.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── prompts.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       └── golden_dataset/
│
├── data/
│   └── chroma_db/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env.example
├── requirements.txt
├── README.md
└── evaluate.py
```

---

# Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd finance-rag
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
FMP_API_KEY=your_fmp_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Never commit the `.env` file to Git.

---

# Tesseract Installation

Tesseract is required for:

* Image OCR
* Scanned PDF OCR

Verify that Tesseract is available:

```bash
tesseract --version
```

If the command is not recognized, install Tesseract and ensure its executable is available in your system PATH.

---

# Ingesting Documents

Place a supported document in the appropriate input directory.

Supported formats:

```text
.pdf
.docx
.png
.jpg
.jpeg
```

Run the ingestion pipeline:

```bash
python ingest.py <path-to-document>
```

Example:

```bash
python ingest.py data/financial_report.pdf
```

The pipeline performs:

```text
File Detection
      ↓
Parsing / OCR
      ↓
Text Cleaning
      ↓
Chunking
      ↓
Embedding
      ↓
ChromaDB Storage
```

---

# Running the Agent

Start the Finance RAG agent:

```bash
python agent.py
```

Example:

```text
You: What was TechNova's revenue in FY2025?

Agent:
According to the uploaded financial report, TechNova's FY2025
revenue was $520 million.
```

---

# Example Queries

## Document Retrieval

```text
What was the company's revenue in FY2025?
```

Expected tool:

```text
search_financial_documents
```

---

## Financial Data

```text
What is Apple's current stock price?
```

Expected tool:

```text
get_stock_quote
```

---

## Company Information

```text
What sector does Apple operate in?
```

Expected tool:

```text
get_company_profile
```

---

## Financial Statements

```text
What was Apple's latest reported revenue?
```

Expected tool:

```text
get_financial_statements
```

---

## Web Search

```text
What are the latest developments regarding Apple's AI strategy?
```

Expected tool:

```text
search_financial_news
```

---

## Multi-Tool Research

```text
Compare the revenue reported in my uploaded financial report
with Apple's latest financial performance.
```

Expected workflow:

```text
Document Retrieval
        +
FMP Financial Statements
        ↓
Context Fusion
        ↓
Draft Answer
        ↓
Self-Critique
        ↓
Final Answer
```

---

# RAGAs Evaluation

Finance RAG includes a standalone evaluation pipeline to measure retrieval and generation quality.

The benchmark evaluates:

### Faithfulness

Measures whether the generated answer is supported by the retrieved context.

### Answer Relevancy

Measures whether the generated answer directly addresses the question.

### Context Precision

Measures whether the retrieved chunks are relevant to the question.

### Context Recall

Measures whether the retrieval system retrieves the information required to answer the question.

Run the evaluation:

```bash
python evaluate.py
```

The evaluation uses a predefined golden dataset containing financial questions and reference answers.

Example evaluation flow:

```text
Golden Dataset
      ↓
Question
      ↓
Retriever
      ↓
Retrieved Context
      ↓
LLM Generation
      ↓
RAGAs Evaluation
      ↓
Metrics
```

The target benchmark is:

```text
RAGAs Score > 0.85
```

The reported score should always come from an actual evaluation run rather than a hardcoded value.

---

# Example Golden Dataset

Example question:

```text
What was TechNova's revenue in FY2025?
```

Reference answer:

```text
TechNova Solutions Ltd. reported $520 million in revenue
during FY2025.
```

The evaluation pipeline compares the generated response against the reference information and retrieved context.

---

# Self-Critique Configuration

The agent uses the following configuration:

```python
QUALITY_THRESHOLD = 0.85
MAX_ITERATIONS = 3
```

The threshold determines whether the generated response should undergo another refinement cycle.

The maximum iteration limit prevents indefinite agent loops.

---

# Example End-to-End Run

```text
User:
Compare TechNova's FY2025 revenue with Apple's latest revenue
and explain the difference.
```

### Step 1 — Agent Reasoning

The LangGraph agent determines that the query requires:

```text
Document Retrieval
+
FMP Financial Statements
```

### Step 2 — Document Retrieval

ChromaDB retrieves relevant chunks from the uploaded report.

```text
TechNova FY2025 Revenue = $520M
```

### Step 3 — FMP

The financial statements tool retrieves Apple's latest available financial data.

### Step 4 — Context Fusion

The system combines:

```text
Uploaded Financial Report
+
FMP Data
```

### Step 5 — Draft

Groq generates an initial response.

### Step 6 — Critique

The draft is evaluated for:

```text
Faithfulness
Relevance
Completeness
Accuracy
```

### Step 7 — Refinement

If the score is below the configured threshold, the response is rewritten.

### Step 8 — Final Answer

The final grounded response is returned to the user.

---

# Design Principles

The project follows several core engineering principles:

### Modular Architecture

Document processing, retrieval, tools, agent orchestration, and evaluation are separated into independent components.

### Tool-Based Agent Design

External capabilities are exposed to the agent as explicit tools.

### Stateful Orchestration

LangGraph maintains the state required for multi-step reasoning and refinement.

### Grounded Generation

The system attempts to generate answers from retrieved documents and tool results rather than relying solely on the model's internal knowledge.

### Independent Evaluation

RAG quality is measured separately using a golden dataset and RAGAs metrics.

---

# Limitations

This project is designed as an AI engineering and research prototype.

Potential limitations include:

* Financial API availability depends on the FMP plan and API limits.
* Web search quality depends on Tavily results.
* OCR accuracy depends on image quality and document formatting.
* Local ChromaDB is intended for development and experimentation.
* LLM-generated answers should not be treated as professional financial advice.

---

# Future Improvements

Potential extensions include:

* Hybrid dense + keyword retrieval
* Reranking models
* Streaming responses
* Improved citation generation
* Larger evaluation datasets
* Additional financial data providers
* Cloud-based vector storage
* Production observability

These are outside the core implementation of the current project.

---

# Disclaimer

Finance RAG is an educational and engineering project.

The system is not intended to provide financial, investment, legal, or professional advice.

Financial information retrieved through external APIs may be delayed, incomplete, or subject to API-provider limitations.

Always verify important financial information against authoritative sources.

---

# Author

**Hari Rama Shankar**

Mechanical Engineering
IIT Madras

---

## Project Summary

Finance RAG demonstrates an end-to-end agentic RAG architecture for financial research:

```text
PDF / DOCX / Image
        ↓
Multimodal Ingestion
        ↓
OCR / Text Extraction
        ↓
Chunking
        ↓
HuggingFace Embeddings
        ↓
ChromaDB
        ↓
LangGraph ReAct Agent
        ↓
 ┌──────┼─────────┐
 ↓      ↓         ↓
RAG    FMP      Tavily
 └──────┼─────────┘
        ↓
Context Fusion
        ↓
Groq LLM
        ↓
Self-Critique
        ↓
Refinement
        ↓
Final Answer

       +

     RAGAs
       ↓
  Benchmarking
```

**Finance RAG combines document intelligence, agentic reasoning, live financial data, web search, and self-refinement into a single financial research workflow.**

````

### One thing I'd change from your original write-up

Be careful with this statement:

> `RAGAs feeds 7 predefined test questions into the pipeline, bypassing the agent`

That's fine **only if your actual `evaluate.py` does exactly that**. Similarly, don't put a fixed `>0.85` result in the README unless you've actually run the benchmark and obtained it.

For your GitHub README, the strongest evidence will be an actual section like:

```text
## Evaluation Results

| Metric | Score |
|---|---:|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.89 |
| Context Precision | 0.87 |
| Context Recall | 0.90 |
| Overall | 0.89 |
````

**but only after Antigravity actually runs the evaluation and produces those numbers.** That will make the `>0.85` resume bullet genuinely defensible in an interview.
