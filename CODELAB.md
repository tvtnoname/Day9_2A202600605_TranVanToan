# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
   - **Trả lời**: LLM được khởi tạo thông qua hàm `get_llm()` trong `common/llm.py` bằng lớp `ChatOpenAI` trỏ tới OpenRouter API endpoint (`https://openrouter.ai/api/v1`) cùng API Key.
2. Message được gửi đến LLM có cấu trúc gì?
   - **Trả lời**: Message gửi đi có cấu trúc là một mảng tuần tự `[SystemMessage, HumanMessage]`.
3. Tại sao cần có `SystemMessage` và `HumanMessage`?
   - **Trả lời**: `SystemMessage` dùng để quy định vai trò của Agent, các quy chế trả lời và độ dài bài viết. `HumanMessage` dùng để chứa câu hỏi thực tế của người dùng. Việc tách biệt giúp LLM hoạt động chính xác và tránh bị tiêm nhiễm mã độc prompt (prompt injection).

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
   - **Trả lời**: Hàm `@tool` được trang trí cho các hàm Python bên ngoài (như `search_legal_knowledge`) để biến chúng thành Structured Tools cho LangChain.
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
   - **Trả lời**: Đây là một list chứa các dictionary, mỗi dictionary là một bản ghi kiến thức gồm `id`, `keywords` (các từ khóa để matching nhanh) và `text` (nội dung chi tiết để nhồi vào context của RAG).
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)
   - **Trả lời**: LLM được gắn tools bằng phương thức `llm.bind_tools(tools)`. Nó sẽ tự động xuất ra JSON schema của các tools gửi kèm theo API payload cho LLM đọc để quyết định gọi.

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
   - **Trả lời**: Hàm này tự động tạo ra một đồ thị LangGraph với chu trình ReAct (suy nghĩ -> hành động -> quan sát) kết nối LLM với các Tools.
2. So sánh với Stage 2: không còn manual tool loop
   - **Trả lời**: Agent tự đưa ra quyết định gọi tool nào, tự gửi input và nhận output từ tool rồi suy nghĩ tiếp cho tới khi có câu trả lời cuối cùng, không cần viết vòng lặp `while/for` thủ công như Stage 2.
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần
   - **Trả lời**: Gọi `graph.ainvoke()` một lần duy nhất với câu hỏi đầu vào, đồ thị sẽ tự chạy toàn bộ các chu trình ẩn bên dưới.

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
   - **Trả lời**: Cấu trúc dữ liệu dùng chung giữa các node của đồ thị. Mỗi node nhận state và trả về các trường cập nhật.
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
   - **Trả lời**: Đây là các Agent xử lý chuyên biệt độc lập trên các khía cạnh khác nhau của câu hỏi.
3. Tìm `Send()` API — dispatch parallel tasks
   - **Trả lời**: Được sử dụng để gửi song song các tác vụ bất đồng bộ đến các agent con (`tax_agent`, `compliance_agent`) dựa vào quyết định định tuyến.
4. Xem `graph.add_node()` và `graph.add_edge()`
   - **Trả lời**: Các hàm dùng để khai báo các Node (bước xử lý) và các Edge (luồng di chuyển tuần tự/điều kiện) cấu thành đồ thị LangGraph.

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
   - **Trả lời**: Nên dùng **Single Agent** khi bài toán đơn giản, phạm vi hẹp, số lượng tools ít. Chuyển sang **Multi-Agent** khi prompt của single agent bị quá tải, cần phân định vai trò hoặc cần thực thi song song các chuyên ngành độc lập để tối ưu hiệu năng.
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
   - **Trả lời**: Chuẩn hóa giao tiếp giữa các Agent thông qua các khái niệm cấp cao như Agent Card (mô tả năng lực), Task/Part (đóng gói kết quả đa phương tiện), quản lý tiến trình bất đồng bộ (Task State) và hỗ trợ theo vết yêu cầu (Trace ID) xuyên suốt.
3. Làm thế nào để prevent infinite delegation loops trong A2A?
   - **Trả lời**: Sử dụng trường `delegation_depth` trong metadata của tin nhắn. Mỗi khi ủy quyền, độ sâu tăng thêm 1. Khi đạt giới hạn `MAX_DELEGATION_DEPTH` (ví dụ = 3), Agent sẽ dừng ủy quyền tiếp.
4. Tại sao cần Registry service? Có thể hardcode URLs không?
   - **Trả lời**: Để khám phá dịch vụ động (Service Discovery). Tránh hardcode URLs vì các Agent có thể đổi IP, Port, hoặc được scale động. Registry giúp hệ thống hoạt động linh hoạt, mềm dẻo.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## **Bài Tập Cộng Điểm:**
Sau khi chạy E2E Stage 5 (test_client.py) trả lời 2 câu hỏi:
- Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
  - **Trả lời**: Latency đo đạc thực tế ban đầu trên hệ thống là **64.15 giây**.
- Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?
  - **Trả lời**:
    1. *Bypass LLM Customer Agent*: Chuyển thẳng request đến Law Agent bằng Python code thay vì chạy qua đồ thị LLM Customer Agent (giảm 2 lượt gọi LLM).
    2. *Fast Keyword Routing*: Sửa node `check_routing` của Law Agent thành khớp từ khóa bằng Python code thay vì gọi LLM (giảm 1 lượt gọi LLM).
    3. *Kết quả*: Số lượt gọi LLM tuần tự giảm từ 6 xuống còn 3 lượt, độ trễ hệ thống giảm xuống dưới 30 giây (giảm hơn 50% Latency).

**Chúc các bạn học tốt! 🚀**
