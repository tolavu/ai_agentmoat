"""
app_multi_persona_demo.py
================================================================================
Ứng dụng Streamlit chuyên biệt trình diễn & giải phẫu chi tiết Phần 3:
BÀI TOÁN ĐA TÁC TỬ & PHẢN BIỆN (MULTI-PERSONA REASONING)
Tích hợp trực tiếp vào mã nguồn dự án: https://github.com/tsownsogut/ai_agentmoat
(Dựa trên kiến trúc gốc của Erik Taylor - eriktaylor/ai-agent-moat)

Cách chạy trong thư mục gốc của repository:
    streamlit run app_multi_persona_demo.py
================================================================================
"""

import os
import time
import inspect
import traceback
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

# Thử nạp biến môi trường từ .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1. CẤU HÌNH TRANG & CSS TÙY CHỈNH CHO TRÌNH DIỄN DEMO
# ==============================================================================
st.set_page_config(
    page_title="Multi-Persona Reasoning Demo | AI Agent Moat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 1.25rem;
        border-left: 6px solid #38bdf8;
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 1.75rem;
        margin-bottom: 0.35rem;
    }
    .main-header p {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
    }
    .persona-card {
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        height: 100%;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .persona-market {
        border-top: 5px solid #10b981;
    }
    .persona-value {
        border-top: 5px solid #3b82f6;
    }
    .persona-devil {
        border-top: 5px solid #ef4444;
    }
    .consensus-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #f59e0b;
        border-left: 6px solid #d97706;
        border-radius: 10px;
        padding: 1.35rem 1.5rem;
        color: #1e293b;
        margin-top: 0.75rem;
    }
    .isolation-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-bottom: 0.5rem;
    }
    .tier-badge-critical {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .tier-badge-normal {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #e2e8f0;
        color: #334155;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# 2. LỚP TÍCH HỢP VỚI SOURCE CODE GỐC (tsownsogut/ai_agentmoat)
# ==============================================================================
REPO_MODULES_STATUS: Dict[str, Any] = {
    "research_agent": False,
    "tools": False,
    "display_utils": False,
    "ResearchAgentClass": None,
    "import_error": None,
}

try:
    import research_agent as repo_research_agent
    if hasattr(repo_research_agent, "ResearchAgent"):
        REPO_MODULES_STATUS["research_agent"] = True
        REPO_MODULES_STATUS["ResearchAgentClass"] = repo_research_agent.ResearchAgent
except Exception as e:
    REPO_MODULES_STATUS["import_error"] = str(e)

try:
    import tools as repo_tools
    REPO_MODULES_STATUS["tools"] = True
except Exception:
    repo_tools = None

try:
    import display_utils as repo_display_utils
    REPO_MODULES_STATUS["display_utils"] = True
except Exception:
    repo_display_utils = None


# ==============================================================================
# 3. DỮ LIỆU CHUẨN (BENCHMARK DATA) TỪ BÀI BÁO & MÃ NGUỒN GỐC (NVIDIA - NVDA)
# ==============================================================================
PERSONA_SPECS = [
    {
        "id": "market_investor",
        "name": "1. Market Investor (Nhà đầu tư Ngắn hạn)",
        "method": "generate_market_outlook()",
        "internal_call": "_run_analysis(company, ticker, market_prompt, query)",
        "color": "🟢",
        "css_class": "persona-market",
        "focus": "Động lượng giá (Momentum), xu hướng ngành, tâm lý đám đông & chất xúc tác ngắn hạn.",
        "core_question": "“Thị trường đang nghĩ gì về cổ phiếu này và dòng tiền ngắn hạn đang dịch chuyển ra sao?”",
        "receives_raw_rag": True,
        "sees_other_personas": False,
        "system_prompt_template": (
            "You are a sharp, short-term Market Investor and momentum trader. "
            "Analyze the provided financial data and retrieved market news for {company} ({ticker}). "
            "Focus strictly on price momentum, sector tailwinds, earnings surprises, institutional/retail sentiment, "
            "and near-term catalysts. Cite sources with dates where available. Ignore deep multi-year value metrics "
            "unless they affect immediate market sentiment."
        ),
    },
    {
        "id": "value_investor",
        "name": "2. Value Investor (Nhà đầu tư Giá trị Dài hạn)",
        "method": "generate_value_analysis()",
        "internal_call": "_run_analysis(company, ticker, value_prompt, query)",
        "color": "🔵",
        "css_class": "persona-value",
        "focus": "Giá trị nội tại (Intrinsic Value), lợi thế cạnh tranh (Economic Moat), hệ số định giá & Biên an toàn (Margin of Safety).",
        "core_question": "“Mức giá hiện tại có hợp lý so với giá trị thực và biên an toàn dài hạn hay không?”",
        "receives_raw_rag": True,
        "sees_other_personas": False,
        "system_prompt_template": (
            "You are a disciplined, long-term Value Investor inspired by Warren Buffett and Benjamin Graham. "
            "Evaluate {company} ({ticker}) based on its durable competitive advantage (economic moat), "
            "balance sheet strength, free cash flow, P/E and valuation multiples, and Margin of Safety. "
            "Determine whether the stock is undervalued, fairly priced, or overvalued relative to fundamentals."
        ),
    },
    {
        "id": "devils_advocate",
        "name": "3. Devil’s Advocate (Tác tử Phản biện & Stress-Test)",
        "method": "generate_devils_advocate_view()",
        "internal_call": "_run_analysis(company, ticker, devil_prompt, query)",
        "color": "🔴",
        "css_class": "persona-devil",
        "focus": "Phản biện luận điểm đầu tư, khai thác tin tiêu cực (issues, concerns, investigation, short interest), rủi ro vĩ mô, pháp lý & thực thi.",
        "core_question": "“Luận điểm đầu tư này có thể sụp đổ vì những rủi ro ẩn giấu hay điểm mù nào?”",
        "receives_raw_rag": True,
        "sees_other_personas": False,
        "system_prompt_template": (
            "You are a skeptical Devil's Advocate risk analyst. Your sole mission is to stress-test the bull case "
            "for {company} ({ticker}) and uncover hidden downside risks. Actively scrutinize macroeconomic headwinds, "
            "regulatory or antitrust investigations, supply-chain bottlenecks, customer concentration, competitive threats, "
            "and valuation compression risks using the retrieved evidence."
        ),
    },
    {
        "id": "lead_analyst",
        "name": "4. Lead Analyst (Tác tử Đồng thuận & Ra quyết định)",
        "method": "generate_final_summary()",
        "internal_call": "synthesis_chain.invoke({company, market_out, value_out, devil_out})",
        "color": "🟡",
        "css_class": "consensus-box",
        "focus": "Tổng hợp, đối chiếu xung đột giữa 3 bản báo cáo độc lập và đưa ra xếp hạng hành động cuối cùng.",
        "core_question": "“Khi cân bằng giữa tâm lý thị trường, giá trị cơ bản và rủi ro phản biện, khuyến nghị hành động tối ưu là gì?”",
        "receives_raw_rag": False,
        "sees_other_personas": True,
        "system_prompt_template": (
            "You are the Lead Investment Analyst chairing an investment committee for {company}. "
            "You are given three independent reports: (1) Market Investor Outlook, (2) Value Investor Analysis, "
            "and (3) Devil's Advocate Critique. Do NOT invent new data. Weigh the bullish momentum, fundamental margin of safety, "
            "and critical risk factors to produce a consolidated Consensus Rating (e.g., Buy, Hold, Neutral with Caution, Sell) "
            "with a clear, balanced justification."
        ),
    },
]

NVDA_BENCHMARK_DATA = {
    "company": "NVIDIA",
    "ticker": "NVDA",
    "metrics": {
        "Current Price": "$138.25",
        "Market Cap": "$3.39T",
        "Trailing P/E": "54.2x",
        "Forward P/E": "34.8x",
        "Diluted EPS": "$2.55",
        "Revenue Growth (YoY)": "+94.0%",
    },
    "rag_chunks": [
        {
            "tier": "Tầng 1: Official News & Analysis",
            "is_critical": False,
            "query_pattern": '"NVIDIA" OR "NVDA" official earnings data center AI demand',
            "source": "Reuters / Financial Times (2025-07-10)",
            "content": (
                "Doanh thu mảng Trung tâm Dữ liệu (Data Center) của NVIDIA tiếp tục tăng trưởng ba chữ số nhờ nhu cầu "
                "triển khai kiến trúc GPU Hopper và lộ trình sản xuất hàng loạt chip Blackwell từ các tập đoàn điện toán đám mây lớn."
            ),
        },
        {
            "tier": "Tầng 2: Critical Reports & Concerns [Từ khóa: issues, concerns, investigation]",
            "is_critical": True,
            "query_pattern": '"NVIDIA" "issues" OR "concerns" OR "investigation" OR "short interest"',
            "source": "Wall Street Journal / Bloomberg Regulatory Brief (2025-07-12)",
            "content": (
                "Cơ quan quản lý chống độc quyền tại Mỹ và EU đang gia tăng giám sát đối với vị thế thống trị của hệ sinh thái phần mềm CUDA "
                "và các thỏa thuận cung ứng GPU. Ngoài ra, giới phân tích cảnh báo rủi ro bền vững chi tiêu vốn (CapEx) của các Hyperscalers "
                "nếu tốc độ thương mại hóa ứng dụng AI đầu cuối chậm hơn kỳ vọng."
            ),
        },
        {
            "tier": "Tầng 2: Critical Reports & Concerns [Từ khóa: issues, execution, supply]",
            "is_critical": True,
            "query_pattern": '"NVDA" "execution" OR "supply chain" OR "margin compression"',
            "source": "Semiconductor Industry Analysis Report (2025-07-14)",
            "content": (
                "Nút thắt đóng gói tiên tiến CoWoS tại TSMC và chi phí chuyển đổi thế hệ kiến trúc có thể tạo áp lực lên biên lợi nhuận gộp "
                "trong ngắn hạn, trong khi các khách hàng lớn như Google, Amazon và Microsoft đang đẩy mạnh chip ASIC tự thiết kế."
            ),
        },
        {
            "tier": "Tầng 3: Retail Investor Sentiment (Reddit / Seeking Alpha)",
            "is_critical": False,
            "query_pattern": 'site:reddit.com OR site:seekingalpha.com "NVDA" sentiment valuation',
            "source": "Seeking Alpha & r/stocks Community Digest (2025-07-15)",
            "content": (
                "Nhà đầu tư cá nhân duy trì tâm lý lạc quan mạnh mẽ về vị thế dẫn đầu hạ tầng AI của NVDA, nhưng bắt đầu tranh luận sôi nổi "
                "về khả năng dịch chuyển nhu cầu từ huấn luyện (Training) sang suy luận (Inference) trong 12–18 tháng tới."
            ),
        },
        {
            "tier": "Tầng 4: Deep Dive & Filings (PyMuPDF Extracted)",
            "is_critical": False,
            "query_pattern": '"NVIDIA" "10-Q" OR "investor presentation" moat CUDA networking',
            "source": "NVIDIA Q1 FY26 Form 10-Q & Investor Deck (2025-06)",
            "content": (
                "Hệ sinh thái kết hợp phần cứng GPU, kết nối tốc độ cao NVLink/InfiniBand và nền tảng phần mềm CUDA Enterprise tạo ra "
                "chi phí chuyển đổi (switching cost) rất cao cho khách hàng doanh nghiệp."
            ),
        },
    ],
    "outputs": {
        "market_investor": (
            "### 🟢 Góc nhìn Market Investor (Động lượng & Tâm lý Thị trường)\n"
            "- **Động lượng tăng trưởng vượt trội:** NVIDIA đang hưởng lợi trực tiếp từ làn sóng đầu tư hạ tầng AI toàn cầu, "
            "với doanh thu Data Center liên tục vượt dự báo Phố Wall [Nguồn: Reuters, 2025-07-10].\n"
            "- **Xu hướng ngành thuận lợi (Sector Tailwinds):** Nhu cầu đối với dòng chip Blackwell và hệ thống mạng NVLink duy trì "
            "trạng thái cung không đủ cầu trong ngắn hạn.\n"
            "- **Tâm lý thị trường:** Dòng tiền tổ chức và nhà đầu tư cá nhân đều nghiêng mạnh về kịch bản tăng trưởng tiếp diễn, "
            "tạo câu chuyện dẫn dắt (growth narrative) hấp dẫn nhất nhóm bán dẫn."
        ),
        "value_investor": (
            "### 🔵 Góc nhìn Value Investor (Giá trị Nội tại & Biên An toàn)\n"
            "- **Hào kinh tế (Economic Moat) vững chắc:** Sự kết hợp giữa phần cứng hiệu năng cao và hệ sinh thái phần mềm CUDA "
            "tạo rào cản gia nhập lớn và chi phí chuyển đổi cao [Nguồn: Form 10-Q, 2025-06].\n"
            "- **Biên an toàn (Margin of Safety) hạn chế:** Ở mức định giá P/E trượt ~54.2x và vốn hóa trên $3.3T, phần lớn kỳ vọng "
            "tăng trưởng 2–3 năm tới đã được phản ánh vào thị giá.\n"
            "- **Kết luận định giá:** Cổ phiếu đang được **định giá hợp lý (fairly priced)** cho một doanh nghiệp xuất sắc, "
            "chứ không còn nằm ở vùng định giá thấp hấp dẫn (deeply undervalued) để giải ngân mạnh."
        ),
        "devils_advocate": (
            "### 🔴 Góc nhìn Devil’s Advocate (Phản biện & Stress-Test Rủi ro)\n"
            "- **Rủi ro vĩ mô & Tính bền vững của CapEx:** Nếu các tập đoàn công nghệ lớn (Hyperscalers) cắt giảm chi tiêu hạ tầng AI "
            "do tỷ suất hoàn vốn (ROI) chưa tương xứng, tăng trưởng đơn hàng của NVDA sẽ giảm tốc đột ngột [Nguồn: WSJ/Bloomberg, 2025-07-12].\n"
            "- **Rủi ro pháp lý & Chống độc quyền:** Các cuộc điều tra của cơ quan quản lý tại Mỹ và EU nhắm vào hệ sinh thái CUDA "
            "có thể hạn chế khả năng định giá và ràng buộc khách hàng.\n"
            "- **Thách thức thực thi & Cạnh tranh dài hạn:** Sự phụ thuộc chuỗi cung ứng đóng gói CoWoS, áp lực biên lợi nhuận khi chuyển đổi "
            "kiến trúc mới, cùng xu hướng khách hàng tự phát triển chip ASIC và sự dịch chuyển sang tác vụ **Inference** trong 12–18 tháng tới "
            "có thể làm xói mòn thị phần độc tôn."
        ),
        "lead_analyst_full": (
            "### 🟡 Kết luận Tổng hợp của Lead Analyst (4-Persona Consensus)\n"
            "**1. Xếp hạng Đồng thuận (Consensus Rating): `NEUTRAL WITH CAUTION` (Trung lập nhưng Thận trọng)**\n\n"
            "**2. Cơ sở tổng hợp & Cân bằng luận điểm:**\n"
            "- Trong khi nhận định của **Market Investor** nhấn mạnh động lượng tăng trưởng tích cực, yếu tố thuận lợi từ xu hướng ngành "
            "và câu chuyện tăng trưởng hấp dẫn, thì phân tích của **Value Investor** lại chỉ ra mức định giá hiện tại chỉ mang lại "
            "**biên an toàn hạn chế** — cổ phiếu đang được định giá tương đối hợp lý chứ chưa bị định giá thấp rõ rệt.\n"
            "- Quan trọng hơn, các phản biện từ **Devil’s Advocate** đã làm nổi bật những trở ngại kinh tế vĩ mô (tính bền vững của CapEx), "
            "rủi ro giám sát pháp lý/chống độc quyền và thách thức thực thi chiến lược có thể tác động đáng kể đến hiệu quả tương lai.\n"
            "- **Khuyến nghị hành động:** Sự giao thoa giữa tâm lý thị trường lạc quan, việc thiếu biên an toàn rõ nét và các rủi ro gia tăng "
            "đòi hỏi nhà đầu tư duy trì thái độ thận trọng. Ghi nhận tiềm năng tăng giá dài hạn nhưng ưu tiên kiểm soát rủi ro do "
            "**chưa có điểm vào lệnh (entry point) đủ hấp dẫn**."
        ),
        "lead_analyst_2persona_ablation": (
            "### ⚠️ Mô phỏng Đối chứng: Kết luận nếu CHỈ dùng 2 Persona (Phiên bản cũ: Market + Value)\n"
            "**1. Xếp hạng Đồng thuận (2-Persona Rating): `MODERATE BUY / ACCUMULATE` (Nghiêng về Mua tích lũy)**\n\n"
            "**2. Phân tích sai lệch khi thiếu Devil’s Advocate:**\n"
            "- Khi loại bỏ tác tử phản biện (`Devil’s Advocate`), Lead Analyst chỉ tiếp nhận động lượng tăng trưởng rất mạnh từ Market Investor "
            "và lợi thế cạnh tranh (CUDA moat) từ Value Investor.\n"
            "- Các rủi ro về điều tra chống độc quyền, sự phụ thuộc vào CapEx của Hyperscalers và nguy cơ dịch chuyển sang chip ASIC/Inference "
            "bị làm mờ hoặc bỏ qua. Hệ quả là mô hình dễ rơi vào **thiên kiến lạc quan (Optimism Bias)** — minh chứng trực tiếp cho lý do "
            "tác giả nâng cấp kiến trúc lên 4 persona."
        ),
    },
}


# ==============================================================================
# 4. HÀM HỖ TRỢ GỌI SOURCE CODE GỐC (tsownsogut/ai_agentmoat)
# ==============================================================================
def run_live_repo_pipeline(
    company: str,
    ticker: str,
    google_api_key: str,
    google_cse_id: str,
    enable_devils_advocate: bool = True,
) -> Dict[str, Any]:
    """
    Kết nối và gọi trực tiếp các thành phần trong repository tsownsogut/ai_agentmoat
    (research_agent.py, tools.py) với cơ chế tự thích ứng chữ ký hàm (signature adaptation).
    """
    logs: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {
        "metrics": {},
        "rag_chunks": [],
        "outputs": {},
        "logs": logs,
    }

    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    if google_cse_id:
        os.environ["GOOGLE_CSE_ID"] = google_cse_id

    # Bước 1: Thu thập chỉ số tài chính bằng yfinance
    t0 = time.perf_counter()
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        mcap = info.get("marketCap")
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        eps = info.get("trailingEps")
        rev_growth = info.get("revenueGrowth")

        results["metrics"] = {
            "Current Price": f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A",
            "Market Cap": f"${mcap / 1e9:,.2f}B" if isinstance(mcap, (int, float)) else "N/A",
            "Trailing P/E": f"{pe:.2f}x" if isinstance(pe, (int, float)) else "N/A",
            "Forward P/E": f"{fpe:.2f}x" if isinstance(fpe, (int, float)) else "N/A",
            "Diluted EPS": f"${eps:.2f}" if isinstance(eps, (int, float)) else "N/A",
            "Revenue Growth": f"{rev_growth * 100:.1f}%" if isinstance(rev_growth, (int, float)) else "N/A",
        }
        financial_summary_str = "\n".join(f"- {k}: {v}" for k, v in results["metrics"].items())
        logs.append({
            "step": "1A. Thu thập chỉ số tài chính (yfinance)",
            "status": "Thành công",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "detail": f"Đã trích xuất {len(results['metrics'])} chỉ số cơ bản cho {ticker}.",
        })
    except Exception as e:
        financial_summary_str = f"Ticker: {ticker}, Company: {company}"
        results["metrics"] = {"Ticker": ticker, "Status": f"Lỗi yfinance: {e}"}
        logs.append({
            "step": "1A. Thu thập chỉ số tài chính (yfinance)",
            "status": "Cảnh báo",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "detail": str(e),
        })

    # Bước 2: Khởi tạo ResearchAgent từ research_agent.py của repo hoặc fallback LangChain
    t1 = time.perf_counter()
    ResearchAgentCls = REPO_MODULES_STATUS.get("ResearchAgentClass")

    if ResearchAgentCls is not None:
        init_sig = inspect.signature(ResearchAgentCls.__init__)
        init_kwargs = {}
        for param_name in init_sig.parameters:
            if param_name == "self":
                continue
            if "api_key" in param_name and google_api_key:
                init_kwargs[param_name] = google_api_key
            elif "cse_id" in param_name and google_cse_id:
                init_kwargs[param_name] = google_cse_id
            elif "company" in param_name:
                init_kwargs[param_name] = company
            elif "ticker" in param_name:
                init_kwargs[param_name] = ticker

        agent = ResearchAgentCls(**init_kwargs)
        logs.append({
            "step": "1B. Khởi tạo ResearchAgent (research_agent.py)",
            "status": "Thành công (Native Repo Class)",
            "latency_ms": round((time.perf_counter() - t1) * 1000, 1),
            "detail": f"Đã nạp lớp ResearchAgent trực tiếp từ mã nguồn repository với tham số: {list(init_kwargs.keys())}.",
        })

        for setup_method in ["build_knowledge_base", "gather_data", "initialize_rag", "prepare_data"]:
            if hasattr(agent, setup_method) and callable(getattr(agent, setup_method)):
                ts = time.perf_counter()
                fn = getattr(agent, setup_method)
                sig = inspect.signature(fn)
                call_args = {}
                for p in sig.parameters:
                    if "company" in p:
                        call_args[p] = company
                    elif "ticker" in p or "symbol" in p:
                        call_args[p] = ticker
                fn(**call_args)
                logs.append({
                    "step": f"1C. Xây dựng kho tri thức RAG ({setup_method})",
                    "status": "Thành công",
                    "latency_ms": round((time.perf_counter() - ts) * 1000, 1),
                    "detail": "Đã lập chỉ mục vector FAISS trong bộ nhớ.",
                })
                break

        def _invoke_agent_method(method_name: str, fallback_args: Dict[str, Any]) -> str:
            fn = getattr(agent, method_name)
            sig = inspect.signature(fn)
            kwargs = {}
            for p in sig.parameters:
                if p in fallback_args:
                    kwargs[p] = fallback_args[p]
                elif "company" in p:
                    kwargs[p] = company
                elif "ticker" in p or "symbol" in p:
                    kwargs[p] = ticker
                elif "financial" in p:
                    kwargs[p] = financial_summary_str
            return fn(**kwargs)

        t_m = time.perf_counter()
        market_out = _invoke_agent_method(
            "generate_market_outlook",
            {"company_name": company, "company": company, "ticker": ticker, "financial_data": financial_summary_str},
        )
        logs.append({
            "step": "2. Persona 1: generate_market_outlook()",
            "status": "Hoàn tất [Ngữ cảnh cô lập]",
            "latency_ms": round((time.perf_counter() - t_m) * 1000, 1),
            "detail": f"Độ dài đầu ra: {len(str(market_out))} ký tự. Không nhận đầu ra từ persona khác.",
        })

        t_v = time.perf_counter()
        value_out = _invoke_agent_method(
            "generate_value_analysis",
            {"company_name": company, "company": company, "ticker": ticker, "financial_data": financial_summary_str},
        )
        logs.append({
            "step": "3. Persona 2: generate_value_analysis()",
            "status": "Hoàn tất [Ngữ cảnh cô lập]",
            "latency_ms": round((time.perf_counter() - t_v) * 1000, 1),
            "detail": f"Độ dài đầu ra: {len(str(value_out))} ký tự. Không nhận đầu ra từ Market Investor.",
        })

        if enable_devils_advocate and hasattr(agent, "generate_devils_advocate_view"):
            t_d = time.perf_counter()
            devil_out = _invoke_agent_method(
                "generate_devils_advocate_view",
                {"company_name": company, "company": company, "ticker": ticker, "financial_data": financial_summary_str},
            )
            logs.append({
                "step": "4. Persona 3: generate_devils_advocate_view()",
                "status": "Hoàn tất [Ngữ cảnh cô lập]",
                "latency_ms": round((time.perf_counter() - t_d) * 1000, 1),
                "detail": f"Độ dài đầu ra: {len(str(devil_out))} ký tự. Đã stress-test luận điểm từ dữ liệu tiêu cực.",
            })
        else:
            devil_out = "[Đã tắt Persona Devil's Advocate trong chế độ kiểm thử đối chứng 2-Persona]"
            logs.append({
                "step": "4. Persona 3: generate_devils_advocate_view()",
                "status": "Đã bỏ qua (Ablation Mode)",
                "latency_ms": 0.0,
                "detail": "Người dùng tắt tác tử phản biện để quan sát sai lệch lạc quan.",
            })

        t_l = time.perf_counter()
        sum_fn = getattr(agent, "generate_final_summary")
        sum_sig = inspect.signature(sum_fn)
        sum_params = [p for p in sum_sig.parameters if p != "self"]
        if len(sum_params) >= 4:
            lead_out = sum_fn(company, market_out, value_out, devil_out)
        else:
            lead_out = sum_fn(company, market_out, value_out)

        logs.append({
            "step": "5. Consensus Agent: generate_final_summary()",
            "status": "Hoàn tất [Tổng hợp bậc cao]",
            "latency_ms": round((time.perf_counter() - t_l) * 1000, 1),
            "detail": "Chỉ nhận 3 bản tóm tắt văn bản; không truy cập trực tiếp dữ liệu thô.",
        })

        results["outputs"] = {
            "market_investor": str(market_out),
            "value_investor": str(value_out),
            "devils_advocate": str(devil_out),
            "lead_analyst_full": str(lead_out),
        }
        return results

    # Fallback độc lập dùng LangChain + Gemini 2.5 Flash (temperature=0.2)
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    parser = StrOutputParser()

    raw_snippets: List[Dict[str, Any]] = []
    try:
        from duckduckgo_search import DDGS
        tier_queries = [
            ("Tầng 1: Official News & Analysis", False, f"{company} {ticker} stock earnings news"),
            ("Tầng 2: Critical Reports & Concerns [issues, concerns, investigation]", True, f"{company} {ticker} issues OR concerns OR investigation OR short interest"),
            ("Tầng 3: Retail Investor Sentiment (Reddit / Seeking Alpha)", False, f"{company} {ticker} site:reddit.com OR site:seekingalpha.com"),
            ("Tầng 4: Deep Dive & Competitive Moat", False, f"{company} {ticker} competitive advantage economic moat report"),
        ]
        with DDGS() as ddgs:
            for tier_label, is_crit, q in tier_queries:
                hits = list(ddgs.text(q, max_results=2))
                for h in hits:
                    raw_snippets.append({
                        "tier": tier_label,
                        "is_critical": is_crit,
                        "query_pattern": q,
                        "source": h.get("title", "Web Source"),
                        "content": h.get("body", ""),
                    })
    except Exception as search_err:
        raw_snippets.append({
            "tier": "Tầng 1: Financial Metrics Context",
            "is_critical": False,
            "query_pattern": "yfinance fallback",
            "source": "Yahoo Finance API",
            "content": financial_summary_str + f" (Search note: {search_err})",
        })

    results["rag_chunks"] = raw_snippets
    combined_rag_context = "\n\n".join(
        f"[{c['tier']}] ({c['source']}): {c['content']}" for c in raw_snippets
    )

    def _run_analysis_standalone(spec: Dict[str, Any]) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", spec["system_prompt_template"]),
            ("human", "Financial Metrics:\n{financial_data}\n\nRetrieved RAG Context:\n{rag_context}\n\nProvide a concise, structured Vietnamese analysis with source citations."),
        ])
        chain = prompt | llm | parser
        return chain.invoke({
            "company": company,
            "ticker": ticker,
            "financial_data": financial_summary_str,
            "rag_context": combined_rag_context,
        })

    t_m = time.perf_counter()
    market_out = _run_analysis_standalone(PERSONA_SPECS[0])
    logs.append({
        "step": "2. Persona 1: generate_market_outlook() [Standalone Gemini 2.5 Flash]",
        "status": "Hoàn tất [Ngữ cảnh cô lập]",
        "latency_ms": round((time.perf_counter() - t_m) * 1000, 1),
        "detail": "temperature=0.2 | Dùng chung RAG context nhưng tách biệt hoàn toàn với Persona 2 & 3.",
    })

    t_v = time.perf_counter()
    value_out = _run_analysis_standalone(PERSONA_SPECS[1])
    logs.append({
        "step": "3. Persona 2: generate_value_analysis() [Standalone Gemini 2.5 Flash]",
        "status": "Hoàn tất [Ngữ cảnh cô lập]",
        "latency_ms": round((time.perf_counter() - t_v) * 1000, 1),
        "detail": "temperature=0.2 | Phân tích giá trị nội tại & Margin of Safety độc lập.",
    })

    if enable_devils_advocate:
        t_d = time.perf_counter()
        devil_out = _run_analysis_standalone(PERSONA_SPECS[2])
        logs.append({
            "step": "4. Persona 3: generate_devils_advocate_view() [Standalone Gemini 2.5 Flash]",
            "status": "Hoàn tất [Ngữ cảnh cô lập]",
            "latency_ms": round((time.perf_counter() - t_d) * 1000, 1),
            "detail": "temperature=0.2 | Khai thác dữ liệu tiêu cực để stress-test luận điểm đầu tư.",
        })
    else:
        devil_out = "[Đã tắt Persona Devil's Advocate trong chế độ đối chứng 2-Persona]"
        logs.append({
            "step": "4. Persona 3: generate_devils_advocate_view()",
            "status": "Đã bỏ qua (Ablation Mode)",
            "latency_ms": 0.0,
            "detail": "Kiểm thử tác động khi thiếu góc nhìn phản biện.",
        })

    t_l = time.perf_counter()
    lead_prompt = ChatPromptTemplate.from_messages([
        ("system", PERSONA_SPECS[3]["system_prompt_template"]),
        ("human", (
            "Company: {company} ({ticker})\n\n"
            "--- REPORT 1: MARKET INVESTOR ---\n{market_out}\n\n"
            "--- REPORT 2: VALUE INVESTOR ---\n{value_out}\n\n"
            "--- REPORT 3: DEVIL'S ADVOCATE ---\n{devil_out}\n\n"
            "Synthesize these perspectives into a final Consensus Rating and Justification in Vietnamese."
        )),
    ])
    lead_out = (lead_prompt | llm | parser).invoke({
        "company": company,
        "ticker": ticker,
        "market_out": market_out,
        "value_out": value_out,
        "devil_out": devil_out,
    })
    logs.append({
        "step": "5. Consensus Agent: generate_final_summary()",
        "status": "Hoàn tất [Tổng hợp bậc cao]",
        "latency_ms": round((time.perf_counter() - t_l) * 1000, 1),
        "detail": "Lead Analyst chỉ nhận đầu ra văn bản của 3 persona, không đọc dữ liệu thô.",
    })

    results["outputs"] = {
        "market_investor": market_out,
        "value_investor": value_out,
        "devils_advocate": devil_out,
        "lead_analyst_full": lead_out,
    }
    return results


# ==============================================================================
# 5. GIAO DIỆN SIDEBAR (ĐIỀU KHIỂN TRÌNH DIỄN & CẤU HÌNH THAM SỐ)
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Bảng điều khiển Demo")
    st.caption("Tích hợp cho repo: `tsownsogut/ai_agentmoat`")

    with st.expander("🔌 Trạng thái tích hợp Mã nguồn Gốc", expanded=True):
        st.write(f"- `research_agent.py`: {'✅ Đã kết nối' if REPO_MODULES_STATUS['research_agent'] else '⚠️ Chế độ độc lập'}")
        st.write(f"- `tools.py`: {'✅ Đã kết nối' if REPO_MODULES_STATUS['tools'] else 'ℹ️ Dự phòng tích hợp'}")
        st.write(f"- `display_utils.py`: {'✅ Đã kết nối' if REPO_MODULES_STATUS['display_utils'] else 'ℹ️ Dự phòng tích hợp'}")
        if not REPO_MODULES_STATUS["research_agent"]:
            st.info(
                "Đặt tệp `app_multi_persona_demo.py` vào thư mục gốc của repo `ai_agentmoat` "
                "để gọi trực tiếp lớp `ResearchAgent`."
            )

    exec_mode = st.radio(
        "Chế độ trình diễn:",
        [
            "🎬 Demo Chuẩn (NVDA Benchmark - Không cần API Key)",
            "⚡ Chạy Thực tế (Live ResearchAgent + Gemini 2.5 Flash)",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎯 Tham số Doanh nghiệp")
    company_input = st.text_input("Tên công ty:", value="NVIDIA")
    ticker_input = st.text_input("Mã cổ phiếu (Ticker):", value="NVDA").upper().strip()

    st.markdown("### 🧪 Thí nghiệm Đối chứng (Ablation)")
    enable_da = st.checkbox(
        "Kích hoạt Persona 3: Devil’s Advocate",
        value=True,
        help="Bỏ chọn để mô phỏng phiên bản cũ chỉ có 2 persona (Market + Value) và quan sát thiên kiến lạc quan.",
    )

    st.markdown("### 🤖 Cấu hình LLM Gốc")
    st.code(
        'model = "gemini-2.5-flash"\n'
        "temperature = 0.2\n"
        'embedding = "all-MiniLM-L6-v2"\n'
        'vector_store = "FAISS (in-memory)"',
        language="python",
    )

    api_key_input = ""
    cse_id_input = ""
    if "Live" in exec_mode:
        api_key_input = st.text_input(
            "GOOGLE_API_KEY:",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
        )
        cse_id_input = st.text_input(
            "GOOGLE_CSE_ID (Tùy chọn):",
            value=os.environ.get("GOOGLE_CSE_ID", ""),
            type="password",
        )


# ==============================================================================
# 6. TIÊU ĐỀ CHÍNH & CHỈ SỐ TÀI CHÍNH ĐẦU VÀO
# ==============================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 Phần 3: Bài toán Đa Tác tử & Phản biện (Multi-Persona Reasoning)</h1>
        <p>
            Trình diễn trực quan cơ chế phân tách vai trò nhận thức, cô lập ngữ cảnh theo chiều ngang,
            khai thác dữ liệu phản biện đa tầng và tổng hợp đồng thuận trên kiến trúc <b>ai_agentmoat</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "demo_state" not in st.session_state:
    st.session_state["demo_state"] = {
        "company": NVDA_BENCHMARK_DATA["company"],
        "ticker": NVDA_BENCHMARK_DATA["ticker"],
        "metrics": NVDA_BENCHMARK_DATA["metrics"],
        "rag_chunks": NVDA_BENCHMARK_DATA["rag_chunks"],
        "outputs": NVDA_BENCHMARK_DATA["outputs"],
        "logs": [
            {
                "step": "1. Thu thập đa tầng & Lập chỉ mục FAISS",
                "status": "Sẵn sàng (NVDA Benchmark)",
                "latency_ms": 1420.0,
                "detail": "4 tầng tìm kiếm (bao gồm từ khóa tiêu cực: issues, concerns, investigation, short interest).",
            },
            {
                "step": "2. generate_market_outlook()",
                "status": "Hoàn tất [Ngữ cảnh cô lập]",
                "latency_ms": 1850.0,
                "detail": "temperature=0.2 | Chỉ đọc RAG context + financial_data.",
            },
            {
                "step": "3. generate_value_analysis()",
                "status": "Hoàn tất [Ngữ cảnh cô lập]",
                "latency_ms": 1920.0,
                "detail": "temperature=0.2 | Không nhìn thấy đầu ra của Market Investor.",
            },
            {
                "step": "4. generate_devils_advocate_view()",
                "status": "Hoàn tất [Ngữ cảnh cô lập]",
                "latency_ms": 2110.0,
                "detail": "temperature=0.2 | Stress-test luận điểm dựa trên tin tiêu cực Tầng 2.",
            },
            {
                "step": "5. generate_final_summary() [Lead Analyst]",
                "status": "Hoàn tất [Tổng hợp bậc cao]",
                "latency_ms": 1640.0,
                "detail": "Tổng hợp 3 văn bản kết luận -> Xếp hạng: Neutral with Caution.",
            },
        ],
    }

# ==============================================================================
# 7. HỆ THỐNG 5 TAB TRÌNH DIỄN CHI TIẾT PHẦN 3
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Luồng Hoạt động & Cô lập Ngữ cảnh",
    "2️⃣ Giải phẫu Mã nguồn & System Prompts",
    "3️⃣ Kho tri thức RAG & Dữ liệu Phản biện",
    "4️⃣ Trình diễn Đa Tác tử & Phản biện (Live Demo)",
    "5️⃣ Đối chứng 2 vs 4 Persona & Môi trường",
])

# ------------------------------------------------------------------------------
# TAB 1: LUỒNG HOẠT ĐỘNG 5 BƯỚC & MA TRẬN CÔ LẬP NGỮ CẢNH
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Tại sao cần kiến trúc Đa Tác tử & Phản biện?")
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.error(
            "**❌ Hạn chế của Mô hình Một câu lệnh (Single-Prompt LLM):**\n"
            "- **Thiên kiến một chiều (Single-Perspective Bias):** Dễ bị cuốn theo tin tức truyền thông chủ đạo tích cực.\n"
            "- **Xu hướng hùa theo (Sycophancy):** Thường tạo ra nhận định trơn tru, quá lạc quan nhưng thiếu cảnh báo rủi ro đuôi.\n"
            "- **Loãng ngữ cảnh (Context Dilution):** Khi yêu cầu vừa tìm cơ hội vừa tìm rủi ro trong cùng một prompt, LLM thường ưu tiên luận điểm chiếm đa số văn bản."
        )
    with col_prob2:
        st.success(
            "**✅ Giải pháp của Kiến trúc Multi-Persona (`ai-agent-moat`):**\n"
            "- **Phân tách vai trò nhận thức:** Tách một bài toán phân tích thành 3 lăng kính độc lập (Động lượng – Giá trị – Phản biện).\n"
            "- **Dùng chung dữ liệu, khác biệt System Prompt:** Cả 3 tác tử đều truy xuất trên cùng một kho FAISS nhưng bị ràng buộc bởi mục tiêu đánh giá đối nghịch.\n"
            "- **Cơ chế phản biện cấu trúc:** Ép hệ thống đối mặt với rủi ro trước khi `Lead Analyst` ra quyết định."
        )

    st.markdown("---")
    st.subheader("Sơ đồ Luồng Xử lý Tuần tự & Rào cản Cô lập Ngữ cảnh (5 Bước)")

    graphviz_code = """
    digraph MultiPersonaFlow {
        rankdir=TB;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=10];

        subgraph cluster_step1 {
            label="BƯỚC 1: Thu thập Đa tầng & Kho Tri thức Dùng chung (Shared RAG)";
            style="rounded,dashed";
            color="#64748b";
            YFinance [label="Yahoo Finance (yfinance)\\nGiá, Vốn hóa, P/E, EPS", fillcolor="#e0f2fe", color="#0284c7"];
            Search4Tier [label="Tìm kiếm Web 4 Tầng\\n(Tầng 2: issues, concerns, investigation, short interest)", fillcolor="#fee2e2", color="#dc2626"];
            FAISS [label="Kho Vector Tạm thời (FAISS + all-MiniLM-L6-v2)\\nTrích dẫn nguồn & ngày xuất bản", fillcolor="#f1f5f9", color="#334155"];
            YFinance -> FAISS;
            Search4Tier -> FAISS;
        }

        subgraph cluster_step234 {
            label="BƯỚC 2 – 4: Thực thi Tuần tự & CÔ LẬP NGỮ CẢNH THEO CHIỀU NGANG (Horizontal Isolation)";
            style="rounded,solid";
            color="#0ea5e9";
            P1 [label="Persona 1: Market Investor\\ngenerate_market_outlook()\\n[Động lượng & Tâm lý ngắn hạn]", fillcolor="#dcfce7", color="#16a34a"];
            P2 [label="Persona 2: Value Investor\\ngenerate_value_analysis()\\n[Giá trị nội tại & Biên an toàn]", fillcolor="#dbeafe", color="#2563eb"];
            P3 [label="Persona 3: Devil's Advocate\\ngenerate_devils_advocate_view()\\n[Stress-test & Rủi ro vĩ mô/pháp lý]", fillcolor="#fee2e2", color="#dc2626"];
        }

        subgraph cluster_step5 {
            label="BƯỚC 5: Tổng hợp Bậc cao (TÁCH BIỆT DỮ LIỆU THÔ)";
            style="rounded,solid";
            color="#d97706";
            Lead [label="Persona 4: Lead Analyst\\ngenerate_final_summary()\\nChỉ nhận 3 văn bản kết luận -> Xếp hạng hành động", fillcolor="#fef3c7", color="#d97706"];
        }

        FAISS -> P1 [label=" _run_analysis()"];
        FAISS -> P2 [label=" _run_analysis()"];
        FAISS -> P3 [label=" _run_analysis()"];

        P1 -> Lead [label=" Văn bản 1"];
        P2 -> Lead [label=" Văn bản 2"];
        P3 -> Lead [label=" Văn bản 3"];
    }
    """
    st.graphviz_chart(graphviz_code, use_container_width=True)

    st.subheader("Ma trận Kiểm soát Quyền Truy cập Ngữ cảnh (Context Isolation Matrix)")
    isolation_df = pd.DataFrame([
        {
            "Tác tử (Persona)": "Market Investor",
            "Phương thức trong ResearchAgent": "generate_market_outlook()",
            "Truy cập Dữ liệu tài chính (yfinance)": "✅ Có",
            "Truy cập Kho vector FAISS (RAG)": "✅ Có (qua _run_analysis)",
            "Nhìn thấy đầu ra Persona khác": "❌ Không (Cô lập hoàn toàn)",
            "Vai trò trong quyết định": "Cung cấp tín hiệu động lượng & tâm lý đám đông",
        },
        {
            "Tác tử (Persona)": "Value Investor",
            "Phương thức trong ResearchAgent": "generate_value_analysis()",
            "Truy cập Dữ liệu tài chính (yfinance)": "✅ Có",
            "Truy cập Kho vector FAISS (RAG)": "✅ Có (qua _run_analysis)",
            "Nhìn thấy đầu ra Persona khác": "❌ Không (Cô lập hoàn toàn)",
            "Vai trò trong quyết định": "Kiểm định định giá thực & Biên an toàn",
        },
        {
            "Tác tử (Persona)": "Devil’s Advocate",
            "Phương thức trong ResearchAgent": "generate_devils_advocate_view()",
            "Truy cập Dữ liệu tài chính (yfinance)": "✅ Có",
            "Truy cập Kho vector FAISS (RAG)": "✅ Có (qua _run_analysis)",
            "Nhìn thấy đầu ra Persona khác": "❌ Không (Cô lập hoàn toàn)",
            "Vai trò trong quyết định": "Phản biện, tìm rủi ro vĩ mô, pháp lý & thực thi",
        },
        {
            "Tác tử (Persona)": "Lead Analyst",
            "Phương thức trong ResearchAgent": "generate_final_summary()",
            "Truy cập Dữ liệu tài chính (yfinance)": "❌ Không trực tiếp",
            "Truy cập Kho vector FAISS (RAG)": "❌ Không trực tiếp",
            "Nhìn thấy đầu ra Persona khác": "✅ Nhận cả 3 bản báo cáo",
            "Vai trò trong quyết định": "Trọng tài tổng hợp -> Xếp hạng cuối cùng",
        },
    ])
    st.dataframe(isolation_df, use_container_width=True, hide_index=True)

    st.info(
        "**💡 Điểm thiết kế đắt giá:** Vì `app.py` gọi tuần tự 3 phương thức phân tích đầu tiên mà **không truyền đầu ra của persona trước vào persona sau**, "
        "hiện tượng *mỏ neo nhận thức (anchoring bias)* giữa các tác tử bị triệt tiêu. Đồng thời, `Lead Analyst` không nhận lại dữ liệu thô "
        "để tránh bị nhiễu bởi hàng nghìn từ tin tức gốc, buộc nó phải phân xử dựa trên các luận điểm đã được chắt lọc."
    )

# ------------------------------------------------------------------------------
# TAB 2: GIẢI PHẪU MÃ NGUỒN & SYSTEM PROMPTS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Cài đặt trong Mã nguồn (`research_agent.py` & `app.py`)")

    col_code_left, col_code_right = st.columns([1.15, 1])

    with col_code_left:
        st.markdown("#### 1. Cấu trúc lớp `ResearchAgent` & Hàm lõi `_run_analysis`")
        if REPO_MODULES_STATUS["research_agent"] and REPO_MODULES_STATUS["ResearchAgentClass"] is not None:
            st.success("Đang hiển thị mã nguồn trực tiếp từ tệp `research_agent.py` trong repository của bạn:")
            try:
                src_text = inspect.getsource(REPO_MODULES_STATUS["ResearchAgentClass"])
                st.code(src_text, language="python")
            except Exception as e:
                st.warning(f"Không thể trích xuất source động: {e}")
        else:
            st.caption("Mô phỏng cấu trúc thực thi chuẩn của `research_agent.py` và `app.py` trong `ai_agentmoat`:")
            reference_code = """# research_agent.py (Kiến trúc chuẩn trong ai-agent-moat)
from langchain_google_genai import ChatGoogleGenerativeAI

class ResearchAgent:
    def __init__(self):
        # Nhiệt độ thấp (0.2) giúp đầu ra ổn định, bám sát dữ liệu RAG
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

    def _run_analysis(self, company_name: str, ticker: str, system_prompt: str, query: str) -> str:
        # Hàm nội bộ dùng chung cho cả 3 persona phân tích.
        # Chạy chuỗi rag_chain.invoke với 2 đầu vào: input (câu hỏi) và financial_data.
        # Sự khác biệt về kết luận đến 100% từ system_prompt, KHÔNG phải từ nguồn dữ liệu khác nhau.
        response = self.rag_chain.invoke({
            "input": query,
            "financial_data": self.financial_data,
            "system_prompt": system_prompt,
        })
        return response["answer"]

    def generate_market_outlook(self, company_name: str, ticker: str) -> str:
        return self._run_analysis(company_name, ticker, MARKET_INVESTOR_PROMPT, ...)

    def generate_value_analysis(self, company_name: str, ticker: str) -> str:
        return self._run_analysis(company_name, ticker, VALUE_INVESTOR_PROMPT, ...)

    def generate_devils_advocate_view(self, company_name: str, ticker: str) -> str:
        return self._run_analysis(company_name, ticker, DEVILS_ADVOCATE_PROMPT, ...)

    def generate_final_summary(
        self, company_name: str, market_outlook: str, value_analysis: str, devils_advocate: str
    ) -> str:
        # Lead Analyst KHÔNG gọi _run_analysis và KHÔNG đọc dữ liệu thô
        return self.consensus_chain.invoke({
            "company": company_name,
            "market_outlook": market_outlook,
            "value_analysis": value_analysis,
            "devils_advocate": devils_advocate,
        })"""
            st.code(reference_code, language="python")

    with col_code_right:
        st.markdown("#### 2. Giải phẫu `System Prompt` của từng Persona")
        selected_persona_name = st.selectbox(
            "Chọn Persona để xem chi tiết cấu trúc Prompt:",
            [p["name"] for p in PERSONA_SPECS],
        )
        selected_spec = next(p for p in PERSONA_SPECS if p["name"] == selected_persona_name)

        st.markdown(f"**Phương thức gọi:** `{selected_spec['method']}`")
        st.markdown(f"**Cơ chế nội bộ:** `{selected_spec['internal_call']}`")
        st.markdown(f"**Trọng tâm đánh giá:** {selected_spec['focus']}")
        st.markdown(f"**Câu hỏi cốt lõi:** {selected_spec['core_question']}")
        st.markdown("**Cấu trúc System Prompt đặc tả vai trò:**")
        st.code(selected_spec["system_prompt_template"], language="text")

        st.warning(
            "**Tại sao đặt `temperature = 0.2`?**\n"
            "Với `temperature = 0.2` trên `gemini-2.5-flash`, mô hình hạn chế tối đa việc tự suy diễn ngoài ngữ cảnh RAG. "
            "Nhờ đó, khi cùng đọc một tập dữ liệu tài chính, `Market Investor` và `Devil's Advocate` đưa ra kết luận trái ngược nhau "
            "là do **ràng buộc vai trò trong System Prompt**, chứ không phải do nhiễu ngẫu nhiên khi sinh từ."
        )

# ------------------------------------------------------------------------------
# TAB 3: KHO TRI THỨC RAG & DỮ LIỆU PHẢN BIỆN
# ------------------------------------------------------------------------------
with tab3:
    state = st.session_state["demo_state"]
    st.subheader(f"Ngữ cảnh Đầu vào Dùng chung cho `{state['company']} ({state['ticker']})`")

    st.markdown("#### 1. Dữ liệu Định lượng (`yfinance` API)")
    m_cols = st.columns(len(state["metrics"]))
    for idx, (k, v) in enumerate(state["metrics"].items()):
        m_cols[idx].metric(label=k, value=v)

    st.markdown("---")
    st.markdown("#### 2. Kho Tri thức Tạm thời trong Bộ nhớ (`FAISS` + Chiến lược Tìm kiếm 4 Tầng)")
    st.write(
        "Điểm mấu chốt giúp tác tử **Devil’s Advocate** có đủ bằng chứng phản biện (thay vì nói chung chung) "
        "nằm ở **Tầng 2 của chiến lược thu thập dữ liệu**: hệ thống chủ động truy vấn các từ khóa tiêu cực "
        'như `"issues"`, `"concerns"`, `"investigation"`, `"short interest"` để nạp sẵn thông tin bất lợi vào FAISS.'
    )

    only_critical = st.checkbox("🔍 Chỉ lọc các tài liệu phản biện / rủi ro (Tầng 2)", value=False)

    chunks_to_show = state["rag_chunks"]
    if only_critical:
        chunks_to_show = [c for c in chunks_to_show if c.get("is_critical")]

    for idx, chunk in enumerate(chunks_to_show, start=1):
        badge_html = (
            '<span class="tier-badge-critical">⚠️ DỮ LIỆU PHẢN BIỆN CHỦ ĐỘNG</span>'
            if chunk.get("is_critical")
            else '<span class="tier-badge-normal">📄 NGỮ CẢNH THỊ TRƯỜNG / CƠ BẢN</span>'
        )
        with st.container(border=True):
            st.markdown(f"**Chunk #{idx} — {chunk['tier']}** &nbsp; {badge_html}", unsafe_allow_html=True)
            st.caption(f"**Mẫu truy vấn tìm kiếm:** `{chunk['query_pattern']}`  |  **Nguồn trích dẫn:** {chunk['source']}")
            st.write(chunk["content"])

# ------------------------------------------------------------------------------
# TAB 4: TRÌNH DIỄN ĐA TÁC TỬ & PHẢN BIỆN (LIVE DEMO)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Trình diễn Quy trình Thực thi & Đối thoại Phản biện Đa Tác tử")

    col_btn, col_info = st.columns([1.2, 2.5])
    with col_btn:
        run_clicked = st.button(
            "🚀 Chạy Phân tích Đa Tác tử (Run Pipeline)",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            f"Mục tiêu hiện tại: **{company_input} ({ticker_input})** | "
            f"Chế độ: **{exec_mode.split('(')[0].strip()}** | "
            f"Devil's Advocate: **{'BẬT' if enable_da else 'TẮT (Ablation)'}**"
        )

    if run_clicked:
        if "Demo Chuẩn" in exec_mode:
            with st.status("Đang mô phỏng từng bước luồng Multi-Persona Reasoning...", expanded=True) as status:
                st.write("📥 Bước 1: Nạp chỉ số tài chính `yfinance` và kho vector `FAISS` (4 tầng tìm kiếm)...")
                time.sleep(0.35)
                st.write("🟢 Bước 2: Gọi `generate_market_outlook()` — Đánh giá động lượng & tâm lý ngắn hạn...")
                time.sleep(0.35)
                st.write("🔵 Bước 3: Gọi `generate_value_analysis()` — Đánh giá giá trị nội tại & Biên an toàn [Cô lập ngữ cảnh]...")
                time.sleep(0.35)
                if enable_da:
                    st.write("🔴 Bước 4: Gọi `generate_devils_advocate_view()` — Stress-test luận điểm từ dữ liệu tiêu cực [Cô lập ngữ cảnh]...")
                    time.sleep(0.35)
                else:
                    st.write("⏭️ Bước 4: Bỏ qua `generate_devils_advocate_view()` (Chế độ đối chứng 2-Persona)...")
                st.write("🟡 Bước 5: Gọi `generate_final_summary()` — Lead Analyst tổng hợp xếp hạng cuối cùng...")
                time.sleep(0.3)
                status.update(label="✅ Hoàn tất quy trình suy luận 4 Persona!", state="complete", expanded=False)

            st.session_state["demo_state"] = {
                "company": NVDA_BENCHMARK_DATA["company"],
                "ticker": NVDA_BENCHMARK_DATA["ticker"],
                "metrics": NVDA_BENCHMARK_DATA["metrics"],
                "rag_chunks": NVDA_BENCHMARK_DATA["rag_chunks"],
                "outputs": {
                    "market_investor": NVDA_BENCHMARK_DATA["outputs"]["market_investor"],
                    "value_investor": NVDA_BENCHMARK_DATA["outputs"]["value_investor"],
                    "devils_advocate": (
                        NVDA_BENCHMARK_DATA["outputs"]["devils_advocate"]
                        if enable_da
                        else "⚠️ *[Tác tử Devil's Advocate đã bị tắt để thử nghiệm đối chứng 2-Persona]*"
                    ),
                    "lead_analyst_full": (
                        NVDA_BENCHMARK_DATA["outputs"]["lead_analyst_full"]
                        if enable_da
                        else NVDA_BENCHMARK_DATA["outputs"]["lead_analyst_2persona_ablation"]
                    ),
                },
                "logs": st.session_state["demo_state"]["logs"],
            }
        else:
            if not api_key_input and not REPO_MODULES_STATUS["research_agent"]:
                st.error("Vui lòng nhập `GOOGLE_API_KEY` ở thanh bên trái để chạy chế độ Live.")
            else:
                with st.status(f"Đang thực thi ResearchAgent trực tiếp cho {company_input} ({ticker_input})...", expanded=True) as status:
                    try:
                        live_res = run_live_repo_pipeline(
                            company=company_input,
                            ticker=ticker_input,
                            google_api_key=api_key_input,
                            google_cse_id=cse_id_input,
                            enable_devils_advocate=enable_da,
                        )
                        st.session_state["demo_state"] = {
                            "company": company_input,
                            "ticker": ticker_input,
                            "metrics": live_res["metrics"],
                            "rag_chunks": live_res["rag_chunks"] or NVDA_BENCHMARK_DATA["rag_chunks"],
                            "outputs": live_res["outputs"],
                            "logs": live_res["logs"],
                        }
                        status.update(label="✅ Hoàn tất phân tích thực tế!", state="complete", expanded=False)
                    except Exception as run_err:
                        status.update(label="❌ Lỗi khi thực thi pipeline", state="error", expanded=True)
                        st.error(f"Chi tiết lỗi: {run_err}")
                        st.code(traceback.format_exc())

    cur = st.session_state["demo_state"]

    with st.expander("⏱️ Nhật ký Thực thi & Kiểm chứng Cô lập Ngữ cảnh (Execution Trace)", expanded=False):
        st.dataframe(pd.DataFrame(cur["logs"]), use_container_width=True, hide_index=True)

    st.markdown(f"### Kết quả Phân tích Đa chiều cho **{cur['company']} ({cur['ticker']})**")
    col_m, col_v, col_d = st.columns(3)

    with col_m:
        st.markdown('<div class="persona-card persona-market">', unsafe_allow_html=True)
        st.markdown('<span class="isolation-badge">Ngữ cảnh cô lập #1</span>', unsafe_allow_html=True)
        st.markdown(cur["outputs"]["market_investor"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_v:
        st.markdown('<div class="persona-card persona-value">', unsafe_allow_html=True)
        st.markdown('<span class="isolation-badge">Ngữ cảnh cô lập #2</span>', unsafe_allow_html=True)
        st.markdown(cur["outputs"]["value_investor"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="persona-card persona-devil">', unsafe_allow_html=True)
        st.markdown('<span class="isolation-badge">Ngữ cảnh cô lập #3 (Phản biện)</span>', unsafe_allow_html=True)
        st.markdown(cur["outputs"]["devils_advocate"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="consensus-box">', unsafe_allow_html=True)
    st.markdown('<span class="isolation-badge">Tầng Tổng hợp Độc lập (Chỉ đọc 3 văn bản trên)</span>', unsafe_allow_html=True)
    st.markdown(cur["outputs"]["lead_analyst_full"])
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 5: ĐỐI CHỨNG 2 VS 4 PERSONA & SO SÁNH MÔI TRƯỜNG TRIỂN KHAI
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("1. Tiến hóa Kiến trúc: Từ 2 Persona lên 4 Persona")
    evo_df = pd.DataFrame([
        {
            "Tiêu chí": "Số lượng tác tử",
            "Phiên bản V1 (Khởi điểm)": "2 Persona (Market Investor, Value Investor)",
            "Phiên bản Mới nhất (Hiện tại)": "4 Persona (+ Devil’s Advocate, + Lead Analyst)",
        },
        {
            "Tiêu chí": "Cơ chế phản biện rủi ro",
            "Phiên bản V1 (Khởi điểm)": "Phụ thuộc ngẫu nhiên vào việc Value Investor có nhắc tới rủi ro hay không",
            "Phiên bản Mới nhất (Hiện tại)": "Bắt buộc stress-test qua tác tử Devil’s Advocate + tìm kiếm từ khóa tiêu cực",
        },
        {
            "Tiêu chí": "Cơ chế tổng hợp quyết định",
            "Phiên bản V1 (Khởi điểm)": "Người dùng tự đọc 2 nhận định rời rạc và tự kết luận",
            "Phiên bản Mới nhất (Hiện tại)": "Lead Analyst tổng hợp thành 1 xếp hạng hành động thống nhất (vd: Neutral with Caution)",
        },
        {
            "Tiêu chí": "Tính kiểm chứng (Auditability)",
            "Phiên bản V1 (Khởi điểm)": "Chưa chuẩn hóa trích dẫn nguồn dữ liệu",
            "Phiên bản Mới nhất (Hiện tại)": "Tích hợp trích dẫn nguồn RAG kèm ngày xuất bản",
        },
    ])
    st.dataframe(evo_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("2. Ảnh hưởng của Môi trường Triển khai tới Kết quả Đa Tác tử")
    st.write(
        "Như tác giả Erik Taylor lưu ý, cùng một kiến trúc 4 Persona nhưng khi chạy trên **Google Colab** "
        "so với bản triển khai công khai trên **Streamlit Community Cloud** sẽ cho kết quả khác nhau:"
    )
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.info(
            "**🔬 Bản Nghiên cứu trên Google Colab**\n"
            "- **Công cụ tìm kiếm:** Google Custom Search API (`GOOGLE_CSE_ID` + `GOOGLE_API_KEY`).\n"
            "- **Đặc điểm ngữ cảnh RAG:** Độ phủ tin tức tài chính chính thống và báo cáo phân tích sâu cao, "
            "giúp `Devil's Advocate` trích xuất được các rủi ro pháp lý và vĩ mô sắc nét.\n"
            "- **Kết quả minh họa NVDA:** `Neutral with Caution`."
        )
    with env_col2:
        st.warning(
            "**☁️ Bản Công khai trên Streamlit Community Cloud**\n"
            "- **Công cụ tìm kiếm:** DuckDuckGo Search (`duckduckgo-search`, không cần cấu hình CSE ID).\n"
            "- **Đặc điểm ngữ cảnh RAG:** Tập kết quả trả về và thứ hạng tài liệu khác biệt so với Google Search, "
            "dẫn tới sự thay đổi trong trọng số lập luận của từng persona dù giữ nguyên `temperature = 0.2`."
        )
