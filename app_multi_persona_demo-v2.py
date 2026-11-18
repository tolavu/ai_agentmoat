"""
app_multi_persona_demo-v2.py
Ứng dụng Demo trực quan Quy trình Suy luận Đa Tác tử & Phản biện (4 Personas)
Tự động dò tìm model Gemini khả dụng theo API Key (tương thích mọi đời Gemini 2.5 / 3.x).
"""

import os
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN GỌN NHẸ
# ==============================================================================
st.set_page_config(
    page_title="Multi-Persona Agent Demo",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    <style>
        .top-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 1.1rem 1.5rem;
            border-radius: 10px;
            color: #f8fafc;
            margin-bottom: 1rem;
            border-left: 5px solid #38bdf8;
        }
        .top-banner h2 {
            color: #f8fafc;
            margin: 0 0 0.25rem 0;
            font-size: 1.45rem;
        }
        .top-banner p {
            color: #cbd5e1;
            margin: 0;
            font-size: 0.9rem;
        }
        .step-box {
            padding: 0.65rem 0.9rem;
            border-radius: 8px;
            margin-bottom: 0.45rem;
            border: 1px solid #cbd5e1;
            background-color: #f8fafc;
            color: #0f172a;
            font-size: 0.92rem;
        }
        .step-running {
            border-left: 5px solid #0284c7;
            background-color: #e0f2fe;
            font-weight: 600;
        }
        .step-done {
            border-left: 5px solid #16a34a;
            background-color: #f0fdf4;
        }
        .step-wait {
            border-left: 5px solid #94a3b8;
            color: #475569;
        }
        .persona-card {
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            border: 1px solid #cbd5e1;
            background-color: #ffffff;
            color: #0f172a;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        .card-market { border-top: 5px solid #10b981; }
        .card-value { border-top: 5px solid #3b82f6; }
        .card-devil { border-top: 5px solid #ef4444; }
        .card-lead {
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            border: 1px solid #f59e0b;
            border-left: 6px solid #d97706;
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            color: #0f172a;
            margin-top: 0.6rem;
        }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            background-color: #e2e8f0;
            color: #1e293b;
            margin-bottom: 0.45rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SYSTEM PROMPTS CHO 4 PERSONA (THEO KIẾN TRÚC AI-AGENT-MOAT)
# ==============================================================================
PROMPTS = {
    "market": (
        "You are a sharp, short-term Market Investor and momentum trader. "
        "Analyze the financial metrics and retrieved market news for {company} ({ticker}). "
        "Focus strictly on price momentum, sector tailwinds, earnings surprises, institutional/retail sentiment, "
        "and near-term catalysts. Cite sources where available. Answer concisely in Vietnamese using bullet points."
    ),
    "value": (
        "You are a disciplined, long-term Value Investor inspired by Warren Buffett and Benjamin Graham. "
        "Evaluate {company} ({ticker}) based on its durable competitive advantage (economic moat), "
        "balance sheet strength, valuation multiples (P/E, market cap), and Margin of Safety. "
        "Determine whether the stock is undervalued, fairly priced, or overvalued. Answer concisely in Vietnamese using bullet points."
    ),
    "devil": (
        "You are a skeptical Devil's Advocate risk analyst. Your sole mission is to stress-test the bull case "
        "for {company} ({ticker}) and uncover hidden downside risks. Actively scrutinize macroeconomic headwinds, "
        "regulatory/antitrust investigations, supply-chain bottlenecks, customer concentration, and competitive threats "
        "using the retrieved evidence. Answer concisely in Vietnamese using bullet points."
    ),
    "lead": (
        "You are the Lead Investment Analyst chairing an investment committee for {company} ({ticker}). "
        "You are given three independent reports: (1) Market Investor Outlook, (2) Value Investor Analysis, "
        "and (3) Devil's Advocate Critique. Do NOT invent new data. Weigh bullish momentum, fundamental margin of safety, "
        "and critical risk factors to produce a consolidated Consensus Rating (e.g., Buy, Hold, Neutral with Caution, Sell) "
        "with a clear, balanced justification in Vietnamese."
    ),
}

# ==============================================================================
# 3. DỮ LIỆU MẪU CHUẨN (NVDA BENCHMARK - CHẠY KHÔNG CẦN API KEY)
# ==============================================================================
BENCHMARK_NVDA = {
    "company": "NVIDIA",
    "ticker": "NVDA",
    "metrics": {
        "Giá hiện tại": "$138.25",
        "Vốn hóa": "$3.39T",
        "Trailing P/E": "54.2x",
        "Forward P/E": "34.8x",
        "EPS": "$2.55",
        "Tăng trưởng DT": "+94.0%",
    },
    "sources": [
        "[Tầng 1 - Tin chính thống | Reuters] Doanh thu Data Center của NVIDIA tăng trưởng ba chữ số nhờ nhu cầu chip Hopper và lộ trình Blackwell.",
        "[Tầng 2 - Dữ liệu phản biện (issues/investigation) | WSJ & Bloomberg] Cơ quan chống độc quyền Mỹ và EU gia tăng giám sát hệ sinh thái CUDA; lo ngại tính bền vững CapEx của các Hyperscalers.",
        "[Tầng 2 - Dữ liệu phản biện (supply/concerns) | Semiconductor Brief] Nút thắt đóng gói CoWoS tại TSMC và xu hướng khách hàng lớn (Google, Amazon, Microsoft) tự phát triển chip ASIC.",
        "[Tầng 3 - Tâm lý nhà đầu tư | Seeking Alpha & Reddit] Tâm lý lạc quan ngắn hạn nhưng tranh luận về sự dịch chuyển nhu cầu từ Training sang Inference trong 12–18 tháng tới.",
    ],
    "market": (
        "### 🟢 1. Market Investor (Động lượng & Tâm lý Ngắn hạn)\n"
        "- **Động lượng tăng trưởng vượt trội:** Doanh thu mảng Trung tâm Dữ liệu (Data Center) liên tục vượt dự báo Phố Wall nhờ làn sóng đầu tư hạ tầng AI [Nguồn: Reuters].\n"
        "- **Chất xúc tác ngắn hạn:** Dòng chip kiến trúc Blackwell duy trì trạng thái cung không đủ cầu từ các tập đoàn điện toán đám mây lớn.\n"
        "- **Dòng tiền & Tâm lý:** Dòng tiền tổ chức và nhà đầu tư cá nhân đều nghiêng mạnh về kịch bản tăng trưởng tiếp diễn."
    ),
    "value": (
        "### 🔵 2. Value Investor (Giá trị Nội tại & Biên An toàn)\n"
        "- **Hào kinh tế (Economic Moat) vững chắc:** Sự kết hợp giữa GPU hiệu năng cao, kết nối NVLink và nền tảng phần mềm CUDA tạo chi phí chuyển đổi rất lớn.\n"
        "- **Biên an toàn (Margin of Safety) hạn chế:** Ở mức định giá P/E trượt ~54.2x và vốn hóa $3.39T, phần lớn kỳ vọng tăng trưởng 2–3 năm tới đã phản ánh vào thị giá.\n"
        "- **Kết luận định giá:** Cổ phiếu đang ở vùng **định giá hợp lý (Fairly Priced)**, chưa có biên an toàn đủ rộng để giải ngân giá trị."
    ),
    "devil": (
        "### 🔴 3. Devil’s Advocate (Phản biện & Stress-Test Rủi ro)\n"
        "- **Rủi ro vĩ mô & CapEx:** Nếu các Hyperscalers cắt giảm chi tiêu hạ tầng AI do tỷ suất hoàn vốn (ROI) chậm, đơn hàng của NVDA sẽ giảm tốc đột ngột [Nguồn: WSJ/Bloomberg].\n"
        "- **Rủi ro pháp lý & Chống độc quyền:** Các cuộc điều tra của Mỹ và EU nhắm vào hệ sinh thái CUDA có thể hạn chế khả năng ràng buộc khách hàng.\n"
        "- **Cạnh tranh dài hạn:** Nút thắt chuỗi cung ứng CoWoS, chip ASIC tự thiết kế của Big Tech và sự dịch chuyển sang tác vụ **Inference** đe dọa thị phần độc tôn."
    ),
    "lead": (
        "### 🟡 4. Lead Analyst — Kết luận Đồng thuận Cuối cùng\n"
        "**Xếp hạng Đồng thuận (Consensus Rating): `NEUTRAL WITH CAUTION` (Trung lập nhưng Thận trọng)**\n\n"
        "- **Tổng hợp đối chiếu:** **Market Investor** ghi nhận động lượng tăng trưởng và câu chuyện dẫn dắt ngành AI rất mạnh, nhưng **Value Investor** chỉ ra mức định giá hiện tại mang lại biên an toàn hạn chế.\n"
        "- **Tác động từ phản biện:** Các cảnh báo của **Devil’s Advocate** về giám sát chống độc quyền, tính bền vững của CapEx và cạnh tranh chip ASIC làm gia tăng rủi ro điều chỉnh nếu tăng trưởng chậm lại.\n"
        "- **Khuyến nghị hành động:** Duy trì vị thế quan sát thận trọng; chưa phải điểm vào lệnh tối ưu cho vị thế mua mới do thiếu biên an toàn."
    ),
}

# ==============================================================================
# 4. HÀM DÒ TÌM MODEL KHẢ DỤNG & GỌI GEMINI API (KHÔNG BAO GIỜ LỖI 404 MODEL)
# ==============================================================================
def discover_gemini_models(api_key: str) -> List[str]:
    """
    Hỏi trực tiếp Google API xem API Key hiện tại được cấp quyền dùng những model nào.
    Sắp xếp ưu tiên các bản Flash (2.5 -> 3.x -> các bản Flash/Pro khác).
    """
    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API Key bị từ chối khi kiểm tra danh sách model (HTTP {http_err.code}): {err_body}")
    except Exception as e:
        raise RuntimeError(f"Không thể kết nối tới máy chủ Google API: {e}")

    valid_models: List[str] = []
    for m in data.get("models", []):
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue
        raw_name = m.get("name", "")
        short_name = raw_name.replace("models/", "")
        lower_n = short_name.lower()
        if any(skip in lower_n for skip in ["embedding", "imagen", "veo", "aqa", "tts", "robotics", "computer-use"]):
            continue
        valid_models.append(short_name)

    if not valid_models:
        raise RuntimeError("API Key hợp lệ nhưng không có model generateContent nào được kích hoạt trên Project này.")

    def _priority(name: str) -> Tuple[int, str]:
        ln = name.lower()
        if ln == "gemini-2.5-flash":
            return (0, ln)
        if "2.5-flash" in ln and "lite" not in ln:
            return (1, ln)
        if "flash" in ln and "lite" not in ln and "exp" not in ln:
            return (2, ln)
        if "flash" in ln:
            return (3, ln)
        if "pro" in ln:
            return (4, ln)
        return (5, ln)

    valid_models.sort(key=_priority)
    return valid_models


def fetch_financial_and_rag(company: str, ticker: str) -> Tuple[Dict[str, str], List[str]]:
    """Bước 1: Lấy chỉ số yfinance và tìm kiếm tin tức đa tầng (kèm từ khóa phản biện)."""
    metrics = {}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        mcap = info.get("marketCap")
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        eps = info.get("trailingEps")
        rev = info.get("revenueGrowth")
        metrics = {
            "Giá hiện tại": f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A",
            "Vốn hóa": f"${mcap / 1e9:,.2f}B" if isinstance(mcap, (int, float)) else "N/A",
            "Trailing P/E": f"{pe:.2f}x" if isinstance(pe, (int, float)) else "N/A",
            "Forward P/E": f"{fpe:.2f}x" if isinstance(fpe, (int, float)) else "N/A",
            "EPS": f"${eps:.2f}" if isinstance(eps, (int, float)) else "N/A",
            "Tăng trưởng DT": f"{rev * 100:.1f}%" if isinstance(rev, (int, float)) else "N/A",
        }
    except Exception as e:
        metrics = {"Mã CP": ticker, "Công ty": company, "Ghi chú": f"Dùng ngữ cảnh LLM ({e})"}

    snippets: List[str] = []
    try:
        from duckduckgo_search import DDGS
        queries = [
            ("Tầng 1 - Tin chính thống", f"{company} {ticker} stock earnings revenue news"),
            ("Tầng 2 - Dữ liệu phản biện (issues/investigation)", f"{company} {ticker} issues OR concerns OR investigation OR short interest"),
            ("Tầng 3 - Tâm lý thị trường", f"{company} {ticker} valuation sentiment analyst"),
        ]
        with DDGS() as ddgs:
            for tier_name, q in queries:
                for hit in ddgs.text(q, max_results=2):
                    snippets.append(f"[{tier_name} | {hit.get('title', 'Web')}] {hit.get('body', '')}")
    except Exception:
        snippets.append(
            f"[Ngữ cảnh định lượng] Dữ liệu tài chính {company} ({ticker}): "
            + ", ".join(f"{k}: {v}" for k, v in metrics.items())
        )

    if not snippets:
        snippets.append(
            f"[Ngữ cảnh định lượng] Dữ liệu tài chính {company} ({ticker}): "
            + ", ".join(f"{k}: {v}" for k, v in metrics.items())
        )

    return metrics, snippets


def call_gemini_rest(api_key: str, model_candidates: List[str], system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """
    Gọi thẳng Gemini REST API theo danh sách model thực tế của API Key.
    Trả về (nội_dung_trả_lời, tên_model_đã_chạy_thành_công).
    """
    clean_key = api_key.strip()
    last_error = ""

    for model_name in model_candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = f"Model {model_name} không trả về nội dung (có thể bị bộ lọc an toàn chặn)."
                    continue
                text = candidates[0]["content"]["parts"][0]["text"]
                return text, model_name
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore")
            last_error = f"[{model_name}] HTTP {http_err.code}: {err_body}"
            if http_err.code in (404, 429, 503):
                continue
            raise RuntimeError(last_error)
        except Exception as exc:
            last_error = f"[{model_name}] Lỗi kết nối: {exc}"
            continue

    raise RuntimeError(f"Đã thử các model {model_candidates[:4]} nhưng đều thất bại. Chi tiết lỗi cuối: {last_error}")


def render_pipeline_tracker(placeholder, step_states: List[Dict[str, str]]):
    """Vẽ bảng trạng thái 5 bước theo thời gian thực."""
    html_parts = ["<div style='margin: 0.6rem 0 1rem 0;'>"]
    for s in step_states:
        status = s["state"]
        if status == "running":
            cls = "step-box step-running"
            icon = "⏳ ĐANG CHẠY..."
        elif status == "done":
            cls = "step-box step-done"
            icon = f"✅ HOÀN TẤT ({s.get('time', '')})"
        else:
            cls = "step-box step-wait"
            icon = "⏸️ Chờ tới lượt"
        html_parts.append(
            f"<div class='{cls}'><b>{s['title']}</b> — <span>{icon}</span>"
            f"<br/><small>{s['desc']}</small></div>"
        )
    html_parts.append("</div>")
    placeholder.markdown("".join(html_parts), unsafe_allow_html=True)


# ==============================================================================
# 5. GIAO DIỆN CHÍNH (NHẬP THÔNG SỐ -> BẤM CHẠY -> XEM TIẾN TRÌNH & 4 PERSONA)
# ==============================================================================
st.markdown(
    """
    <div class="top-banner">
        <h2>🧠 Demo Hệ thống Đa Tác tử & Phản biện (4-Persona Reasoning)</h2>
        <p>Tự động nhận diện model Gemini khả dụng trên API Key của bạn (2.5 / 3.x), theo dõi tiến trình 5 bước và xuất báo cáo 4 Persona.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

default_api_key = os.environ.get("GOOGLE_API_KEY", "")
if not default_api_key:
    try:
        default_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        default_api_key = ""

c_in1, c_in2, c_in3, c_in4 = st.columns([1.1, 0.7, 1.8, 1.1])
with c_in1:
    company_input = st.text_input("Tên công ty:", value="NVIDIA", placeholder="VD: NVIDIA, Apple...")
with c_in2:
    ticker_input = st.text_input("Mã cổ phiếu:", value="NVDA", placeholder="VD: NVDA, AAPL...").upper().strip()
with c_in3:
    api_key_input = st.text_input(
        "Google API Key (Bắt buộc khi chạy Live):",
        value=default_api_key,
        type="password",
        placeholder="Dán khóa AIzaSy... vào đây",
    )
with c_in4:
    custom_model_input = st.text_input(
        "Model ưu tiên (Tùy chọn):",
        value="Tự động dò từ Key",
        help="Để mặc định để hệ thống tự lấy model đang bật trên API Key của bạn (VD: gemini-2.5-flash hoặc gemini-3-flash-preview).",
    )

b_col1, b_col2 = st.columns([1.4, 1])
with b_col1:
    run_live_btn = st.button(
        "🚀 CHẠY PHÂN TÍCH THỰC TẾ (LIVE 4 PERSONAS)",
        type="primary",
        use_container_width=True,
    )
with b_col2:
    run_demo_btn = st.button(
        "▶️ CHẠY MẪU NVDA (KHÔNG CẦN API KEY)",
        use_container_width=True,
    )

st.markdown("---")

tracker_placeholder = st.empty()
metrics_placeholder = st.empty()
rag_placeholder = st.empty()

col_p1, col_p2, col_p3 = st.columns(3)
card1_placeholder = col_p1.empty()
card2_placeholder = col_p2.empty()
card3_placeholder = col_p3.empty()
lead_placeholder = st.empty()

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ==============================================================================
# 6. XỬ LÝ KHI NGƯỜI DÙNG BẤM NÚT CHẠY
# ==============================================================================
if run_live_btn or run_demo_btn:
    is_live = run_live_btn

    if is_live and not api_key_input.strip():
        st.error("⚠️ Bạn chưa nhập **Google API Key**. Hãy dán khóa vào ô phía trên hoặc bấm nút **CHẠY MẪU NVDA** bên cạnh.")
    else:
        steps = [
            {"title": "Bước 1/5: Kiểm tra Model Gemini & Thu thập dữ liệu tài chính + RAG đa tầng", "desc": "Xác thực model khả dụng từ API Key, lấy chỉ số Yahoo Finance và tin tức tích cực + tiêu cực.", "state": "wait"},
            {"title": "Bước 2/5: Persona 1 — Market Investor (generate_market_outlook)", "desc": "Phân tích động lượng giá, xu hướng ngành & tâm lý ngắn hạn [Ngữ cảnh cô lập #1].", "state": "wait"},
            {"title": "Bước 3/5: Persona 2 — Value Investor (generate_value_analysis)", "desc": "Đánh giá giá trị nội tại, lợi thế cạnh tranh (Moat) & Biên an toàn [Ngữ cảnh cô lập #2].", "state": "wait"},
            {"title": "Bước 4/5: Persona 3 — Devil’s Advocate (generate_devils_advocate_view)", "desc": "Khai thác tin tiêu cực để phản biện & stress-test rủi ro [Ngữ cảnh cô lập #3].", "state": "wait"},
            {"title": "Bước 5/5: Persona 4 — Lead Analyst (generate_final_summary)", "desc": "Tổng hợp độc lập 3 bản báo cáo trên và đưa ra xếp hạng hành động cuối cùng.", "state": "wait"},
        ]

        try:
            # ---------------- BƯỚC 1 ----------------
            steps[0]["state"] = "running"
            render_pipeline_tracker(tracker_placeholder, steps)
            t0 = time.perf_counter()

            model_candidates: List[str] = []
            if is_live:
                discovered = discover_gemini_models(api_key_input)
                user_m = custom_model_input.strip()
                if user_m and user_m != "Tự động dò từ Key":
                    model_candidates = [user_m] + [m for m in discovered if m != user_m]
                else:
                    model_candidates = discovered

                cur_company = company_input.strip() or ticker_input
                cur_ticker = ticker_input or "NVDA"
                metrics, snippets = fetch_financial_and_rag(cur_company, cur_ticker)
                steps[0]["desc"] = f"Đã tìm thấy {len(model_candidates)} model trên Key (Ưu tiên: {model_candidates[0]})."
            else:
                time.sleep(0.4)
                cur_company = BENCHMARK_NVDA["company"]
                cur_ticker = BENCHMARK_NVDA["ticker"]
                metrics = BENCHMARK_NVDA["metrics"]
                snippets = BENCHMARK_NVDA["sources"]

            steps[0]["state"] = "done"
            steps[0]["time"] = f"{time.perf_counter() - t0:.2f}s"
            render_pipeline_tracker(tracker_placeholder, steps)

            with metrics_placeholder.container():
                st.markdown(f"#### 📊 Dữ liệu đầu vào dùng chung cho **{cur_company} ({cur_ticker})**")
                m_cols = st.columns(len(metrics))
                for idx, (k, v) in enumerate(metrics.items()):
                    m_cols[idx].metric(k, v)

            with rag_placeholder.expander("📄 Xem các mảnh dữ liệu RAG vừa thu thập (Dùng chung cho 3 Persona)", expanded=False):
                if is_live:
                    st.caption(f"Danh sách model khả dụng trên API Key của bạn: `{', '.join(model_candidates[:8])}`")
                for s in snippets:
                    st.write(f"- {s}")

            fin_text = "\n".join(f"- {k}: {v}" for k, v in metrics.items())
            rag_text = "\n".join(snippets)
            shared_human_prompt = f"Công ty: {cur_company} ({cur_ticker})\n\nChỉ số tài chính:\n{fin_text}\n\nDữ liệu tin tức & báo cáo thu thập được:\n{rag_text}"

            used_model_name = "Benchmark Mode"

            # ---------------- BƯỚC 2: MARKET INVESTOR ----------------
            steps[1]["state"] = "running"
            render_pipeline_tracker(tracker_placeholder, steps)
            t1 = time.perf_counter()

            if is_live:
                raw_m, used_model_name = call_gemini_rest(
                    api_key_input,
                    model_candidates,
                    PROMPTS["market"].format(company=cur_company, ticker=cur_ticker),
                    shared_human_prompt,
                )
                market_out = f"### 🟢 1. Market Investor (Ngắn hạn)\n{raw_m}"
            else:
                time.sleep(0.45)
                market_out = BENCHMARK_NVDA["market"]

            steps[1]["state"] = "done"
            steps[1]["time"] = f"{time.perf_counter() - t1:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            card1_placeholder.markdown(
                f"<div class='persona-card card-market'><span class='badge'>Ngữ cảnh cô lập #1</span>\n\n{market_out}</div>",
                unsafe_allow_html=True,
            )

            # ---------------- BƯỚC 3: VALUE INVESTOR ----------------
            steps[2]["state"] = "running"
            render_pipeline_tracker(tracker_placeholder, steps)
            t2 = time.perf_counter()

            if is_live:
                raw_v, used_model_name = call_gemini_rest(
                    api_key_input,
                    model_candidates,
                    PROMPTS["value"].format(company=cur_company, ticker=cur_ticker),
                    shared_human_prompt,
                )
                value_out = f"### 🔵 2. Value Investor (Dài hạn)\n{raw_v}"
            else:
                time.sleep(0.45)
                value_out = BENCHMARK_NVDA["value"]

            steps[2]["state"] = "done"
            steps[2]["time"] = f"{time.perf_counter() - t2:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            card2_placeholder.markdown(
                f"<div class='persona-card card-value'><span class='badge'>Ngữ cảnh cô lập #2</span>\n\n{value_out}</div>",
                unsafe_allow_html=True,
            )

            # ---------------- BƯỚC 4: DEVIL'S ADVOCATE ----------------
            steps[3]["state"] = "running"
            render_pipeline_tracker(tracker_placeholder, steps)
            t3 = time.perf_counter()

            if is_live:
                raw_d, used_model_name = call_gemini_rest(
                    api_key_input,
                    model_candidates,
                    PROMPTS["devil"].format(company=cur_company, ticker=cur_ticker),
                    shared_human_prompt,
                )
                devil_out = f"### 🔴 3. Devil’s Advocate (Phản biện)\n{raw_d}"
            else:
                time.sleep(0.45)
                devil_out = BENCHMARK_NVDA["devil"]

            steps[3]["state"] = "done"
            steps[3]["time"] = f"{time.perf_counter() - t3:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            card3_placeholder.markdown(
                f"<div class='persona-card card-devil'><span class='badge'>Ngữ cảnh cô lập #3</span>\n\n{devil_out}</div>",
                unsafe_allow_html=True,
            )

            # ---------------- BƯỚC 5: LEAD ANALYST ----------------
            steps[4]["state"] = "running"
            render_pipeline_tracker(tracker_placeholder, steps)
            t4 = time.perf_counter()

            if is_live:
                lead_input = (
                    f"Công ty: {cur_company} ({cur_ticker})\n\n"
                    f"--- BÁO CÁO 1: MARKET INVESTOR ---\n{market_out}\n\n"
                    f"--- BÁO CÁO 2: VALUE INVESTOR ---\n{value_out}\n\n"
                    f"--- BÁO CÁO 3: DEVIL'S ADVOCATE ---\n{devil_out}"
                )
                lead_raw, used_model_name = call_gemini_rest(
                    api_key_input,
                    model_candidates,
                    PROMPTS["lead"].format(company=cur_company, ticker=cur_ticker),
                    lead_input,
                )
                lead_out = f"### 🟡 4. Lead Analyst — Kết luận Đồng thuận Cuối cùng\n{lead_raw}"
            else:
                time.sleep(0.45)
                lead_out = BENCHMARK_NVDA["lead"]

            steps[4]["state"] = "done"
            steps[4]["time"] = f"{time.perf_counter() - t4:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            lead_placeholder.markdown(
                f"<div class='card-lead'><span class='badge'>Tổng hợp từ 3 báo cáo độc lập</span>\n\n{lead_out}</div>",
                unsafe_allow_html=True,
            )

            st.session_state["last_result"] = {
                "steps": steps,
                "company": cur_company,
                "ticker": cur_ticker,
                "metrics": metrics,
                "snippets": snippets,
                "market": market_out,
                "value": value_out,
                "devil": devil_out,
                "lead": lead_out,
            }
            st.success(f"✅ Hoàn tất toàn bộ quy trình 5 bước cho {cur_company} ({cur_ticker}) bằng model `{used_model_name}`!")

        except Exception as exc:
            st.error(f"❌ Quá trình chạy gặp lỗi: {exc}")

elif st.session_state["last_result"] is not None:
    saved = st.session_state["last_result"]
    render_pipeline_tracker(tracker_placeholder, saved["steps"])
    with metrics_placeholder.container():
        st.markdown(f"#### 📊 Dữ liệu đầu vào dùng chung cho **{saved['company']} ({saved['ticker']})**")
        m_cols = st.columns(len(saved["metrics"]))
        for idx, (k, v) in enumerate(saved["metrics"].items()):
            m_cols[idx].metric(k, v)
    with rag_placeholder.expander("📄 Xem các mảnh dữ liệu RAG vừa thu thập (Dùng chung cho 3 Persona)", expanded=False):
        for s in saved["snippets"]:
            st.write(f"- {s}")
    card1_placeholder.markdown(
        f"<div class='persona-card card-market'><span class='badge'>Ngữ cảnh cô lập #1</span>\n\n{saved['market']}</div>",
        unsafe_allow_html=True,
    )
    card2_placeholder.markdown(
        f"<div class='persona-card card-value'><span class='badge'>Ngữ cảnh cô lập #2</span>\n\n{saved['value']}</div>",
        unsafe_allow_html=True,
    )
    card3_placeholder.markdown(
        f"<div class='persona-card card-devil'><span class='badge'>Ngữ cảnh cô lập #3</span>\n\n{saved['devil']}</div>",
        unsafe_allow_html=True,
    )
    lead_placeholder.markdown(
        f"<div class='card-lead'><span class='badge'>Tổng hợp từ 3 báo cáo độc lập</span>\n\n{saved['lead']}</div>",
        unsafe_allow_html=True,
    )