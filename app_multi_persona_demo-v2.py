"""
app_multi_persona_demo-v2.py
Ứng dụng Demo trực quan Quy trình Suy luận Đa Tác tử & Phản biện (4 Personas)
Hiển thị minh bạch toàn bộ Kho Dữ liệu Nguồn (Định lượng + Tin tức 4 tầng RAG)
và truy vết chính xác căn cứ dữ liệu ([FIN-x], [SRC-x]) mà từng Persona dùng để ra quyết định.
"""

import os
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN TRỰC QUAN
# ==============================================================================
st.set_page_config(
    page_title="Multi-Persona Agent Demo | Evidence Traceability",
    page_icon="🧠",
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
        .source-card {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.55rem;
            background-color: #ffffff;
            color: #0f172a;
            font-size: 0.88rem;
        }
        .source-critical {
            border-left: 5px solid #dc2626;
            background-color: #fef2f2;
        }
        .source-normal {
            border-left: 5px solid #0284c7;
            background-color: #f8fafc;
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
        .evidence-box {
            background-color: #f1f5f9;
            border: 1px dashed #64748b;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin-top: 0.85rem;
            font-size: 0.86rem;
            color: #1e293b;
        }
        .evidence-box-devil {
            background-color: #fef2f2;
            border: 1px dashed #dc2626;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin-top: 0.85rem;
            font-size: 0.86rem;
            color: #7f1d1d;
        }
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
        .badge-src {
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 5px;
            font-size: 0.75rem;
            font-weight: 700;
            background-color: #0f172a;
            color: #f8fafc;
            margin-right: 0.35rem;
        }
        .badge-crit {
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 5px;
            font-size: 0.73rem;
            font-weight: 700;
            background-color: #fee2e2;
            color: #b91c1c;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SYSTEM PROMPTS ÉP BUỘC TRÍCH DẪN NGUỒN DỮ LIỆU ([FIN-x], [SRC-x])
# ==============================================================================
CITATION_INSTRUCTION = (
    "\n\nBẮT BUỘC VỀ TRÍCH DẪN NGUỒN DỮ LIỆU:\n"
    "- Mọi nhận định bạn đưa ra PHẢI gắn mã nguồn cụ thể lấy từ danh sách dữ liệu đầu vào (ví dụ: [FIN-1], [SRC-1], [SRC-2]) ngay cuối mỗi gạch đầu dòng.\n"
    "- Đặc biệt: Nếu có chỉ số định lượng hợp lệ (khác N/A) trong [FIN-1] đến [FIN-6], bạn PHẢI sử dụng và trích dẫn các chỉ số phù hợp với vai trò của mình.\n"
    "- Cuối câu trả lời, tạo một mục riêng tên là `📌 Căn cứ dữ liệu đã chọn lọc:` liệt kê ngắn gọn các mã [FIN-x] / [SRC-x] bạn đã dùng và lý do chọn chúng."
)

PROMPTS = {
    "market": (
        "You are a sharp, short-term Market Investor and momentum trader. "
        "Analyze the financial metrics and retrieved market news for {company} ({ticker}). "
        "Focus strictly on price momentum, revenue growth ([FIN-6]), sector tailwinds, earnings surprises, "
        "institutional/retail sentiment, and near-term catalysts. Answer concisely in Vietnamese using bullet points."
        + CITATION_INSTRUCTION
    ),
    "value": (
        "You are a disciplined, long-term Value Investor inspired by Warren Buffett and Benjamin Graham. "
        "Evaluate {company} ({ticker}) based on its durable competitive advantage (economic moat), "
        "valuation multiples ([FIN-1] Price, [FIN-2] Market Cap, [FIN-3] Trailing P/E, [FIN-4] Forward P/E, [FIN-5] EPS), "
        "and Margin of Safety. Determine whether the stock is undervalued, fairly priced, or overvalued. "
        "Answer concisely in Vietnamese using bullet points."
        + CITATION_INSTRUCTION
    ),
    "devil": (
        "You are a skeptical Devil's Advocate risk analyst. Your sole mission is to stress-test the bull case "
        "for {company} ({ticker}) and uncover hidden downside risks. Actively scrutinize valuation stretch ([FIN-3]/[FIN-4]), "
        "macroeconomic headwinds, regulatory/antitrust investigations, supply-chain bottlenecks, and competitive threats "
        "using the retrieved negative/critical evidence. Answer concisely in Vietnamese using bullet points."
        + CITATION_INSTRUCTION
    ),
    "lead": (
        "You are the Lead Investment Analyst chairing an investment committee for {company} ({ticker}). "
        "You are given ONLY three independent reports: (1) Market Investor Outlook, (2) Value Investor Analysis, "
        "and (3) Devil's Advocate Critique. You do NOT have access to raw data. Weigh bullish momentum, fundamental margin of safety, "
        "and critical risk factors to produce a consolidated Consensus Rating (e.g., Buy, Hold, Neutral with Caution, Sell) "
        "with a clear, balanced justification in Vietnamese, referencing how the three personas' evidence conflicts."
    ),
}

# ==============================================================================
# 3. DỮ LIỆU DỰ PHÒNG ĐỊNH LƯỢNG KHI YAHOO FINANCE CHẶN IP CLOUD
# ==============================================================================
FALLBACK_QUOTES: Dict[str, Dict[str, Any]] = {
    "NVDA": {"price": 138.25, "mcap": 3390e9, "pe": 54.2, "fpe": 34.8, "eps": 2.55, "rev": 0.94},
    "AAPL": {"price": 228.50, "mcap": 3470e9, "pe": 34.6, "fpe": 30.2, "eps": 6.60, "rev": 0.061},
    "MSFT": {"price": 425.40, "mcap": 3160e9, "pe": 35.1, "fpe": 30.8, "eps": 12.12, "rev": 0.160},
    "GOOGL": {"price": 176.80, "mcap": 2180e9, "pe": 23.4, "fpe": 20.1, "eps": 7.55, "rev": 0.151},
    "AMZN": {"price": 208.10, "mcap": 2190e9, "pe": 44.5, "fpe": 33.6, "eps": 4.68, "rev": 0.110},
    "META": {"price": 582.30, "mcap": 1470e9, "pe": 27.4, "fpe": 23.1, "eps": 21.25, "rev": 0.189},
    "TSLA": {"price": 320.70, "mcap": 1030e9, "pe": 88.6, "fpe": 96.4, "eps": 3.62, "rev": 0.078},
    "AMD": {"price": 142.60, "mcap": 231e9, "pe": 124.0, "fpe": 28.5, "eps": 1.15, "rev": 0.176},
}

BENCHMARK_NVDA = {
    "company": "NVIDIA",
    "ticker": "NVDA",
    "metrics": [
        {"id": "FIN-1", "label": "Giá hiện tại", "value": "$138.25", "source": "Yahoo Finance (yfinance)"},
        {"id": "FIN-2", "label": "Vốn hóa thị trường", "value": "$3.39T", "source": "Yahoo Finance (yfinance)"},
        {"id": "FIN-3", "label": "Trailing P/E", "value": "54.2x", "source": "Yahoo Finance (yfinance)"},
        {"id": "FIN-4", "label": "Forward P/E", "value": "34.8x", "source": "Yahoo Finance (yfinance)"},
        {"id": "FIN-5", "label": "Diluted EPS", "value": "$2.55", "source": "Yahoo Finance (yfinance)"},
        {"id": "FIN-6", "label": "Tăng trưởng DT (YoY)", "value": "+94.0%", "source": "Yahoo Finance (yfinance)"},
    ],
    "rag_sources": [
        {
            "id": "SRC-1",
            "tier": "Tầng 1: Tin chính thống & Kết quả kinh doanh",
            "is_critical": False,
            "publisher": "Reuters / Financial Times (2025-07-10)",
            "query": '"NVIDIA" OR "NVDA" official earnings data center AI demand',
            "url": "https://www.reuters.com/technology/",
            "snippet": "Doanh thu mảng Trung tâm Dữ liệu (Data Center) của NVIDIA tăng trưởng 94% YoY nhờ nhu cầu triển khai GPU Hopper và đơn đặt hàng chip kiến trúc Blackwell từ các tập đoàn điện toán đám mây lớn.",
        },
        {
            "id": "SRC-2",
            "tier": "Tầng 2: Tin tiêu cực & Pháp lý [Từ khóa: investigation, concerns]",
            "is_critical": True,
            "publisher": "Wall Street Journal / Bloomberg (2025-07-12)",
            "query": '"NVIDIA" "issues" OR "concerns" OR "investigation" OR "short interest"',
            "url": "https://www.bloomberg.com/technology",
            "snippet": "Cơ quan chống độc quyền tại Mỹ (DOJ) và EU gia tăng điều tra đối với vị thế thống trị của hệ sinh thái phần mềm CUDA và các thỏa thuận ràng buộc cung ứng GPU.",
        },
        {
            "id": "SRC-3",
            "tier": "Tầng 2: Rủi ro vĩ mô & Chuỗi cung ứng [Từ khóa: issues, supply, concerns]",
            "is_critical": True,
            "publisher": "Semiconductor Industry Brief (2025-07-14)",
            "query": '"NVDA" "execution" OR "supply chain" OR "concerns"',
            "url": "https://www.wsj.com/tech",
            "snippet": "Nút thắt công nghệ đóng gói CoWoS tại TSMC gây áp lực lên tiến độ giao hàng; đồng thời giới phân tích lo ngại rủi ro bền vững chi tiêu vốn (CapEx) của Hyperscalers và xu hướng Big Tech (Google, Amazon, Microsoft) tự thiết kế chip ASIC.",
        },
        {
            "id": "SRC-4",
            "tier": "Tầng 3: Tâm lý nhà đầu tư cá nhân (Reddit / Seeking Alpha)",
            "is_critical": False,
            "publisher": "Seeking Alpha & r/stocks Digest (2025-07-15)",
            "query": 'site:reddit.com OR site:seekingalpha.com "NVDA" sentiment',
            "url": "https://seekingalpha.com/symbol/NVDA",
            "snippet": "Nhà đầu tư cá nhân duy trì tâm lý lạc quan mạnh mẽ về vị thế dẫn đầu AI của NVDA, nhưng tranh luận sôi nổi về nguy cơ nhu cầu dịch chuyển từ huấn luyện (Training) sang suy luận (Inference) trong 12–18 tháng tới.",
        },
        {
            "id": "SRC-5",
            "tier": "Tầng 4: Báo cáo tài chính & Hồ sơ pháp lý (Form 10-Q / PyMuPDF)",
            "is_critical": False,
            "publisher": "NVIDIA Q1 FY26 Form 10-Q (2025-06)",
            "query": '"NVIDIA" "10-Q" moat CUDA NVLink switching cost',
            "url": "https://investor.nvidia.com/",
            "snippet": "Sự kết hợp đồng bộ giữa phần cứng GPU, giao thức kết nối tốc độ cao NVLink/InfiniBand và nền tảng lập trình CUDA Enterprise tạo ra rào cản chuyển đổi (switching cost) rất cao cho khách hàng doanh nghiệp.",
        },
    ],
    "market": {
        "analysis": (
            "### 🟢 1. Market Investor (Ngắn hạn — Động lượng & Tâm lý)\n"
            "- **Động lượng tăng trưởng doanh thu bùng nổ:** Doanh thu tăng **+94.0% YoY** nhờ mảng Data Center liên tục vượt kỳ vọng Phố Wall `[FIN-6]` `[SRC-1]`.\n"
            "- **Chất xúc tác sản phẩm ngắn hạn:** Lộ trình ra mắt dòng chip kiến trúc **Blackwell** duy trì trạng thái cung không đủ cầu từ các tập đoàn đám mây `[SRC-1]`.\n"
            "- **Tâm lý dòng tiền tích cực:** Cộng đồng nhà đầu tư cá nhân và tổ chức vẫn coi NVDA là cổ phiếu dẫn dắt câu chuyện tăng trưởng AI `[SRC-4]`."
        ),
        "evidence_used": (
            "**📌 Căn cứ dữ liệu Market Investor đã chọn từ kho dùng chung:**\n"
            "- `[FIN-6]` Tăng trưởng doanh thu +94.0% YoY *(chứng minh đà tăng trưởng nóng)*.\n"
            "- `[SRC-1]` Tin chính thống Reuters về đơn hàng chip Hopper & Blackwell *(chất xúc tác ngắn hạn)*.\n"
            "- `[SRC-4]` Dữ liệu tâm lý Seeking Alpha / Reddit *(xác nhận kỳ vọng đám đông đang nghiêng về phe mua)*.\n"
            "- *Bỏ qua:* Các chỉ số định giá dài hạn (`[FIN-3]`) và tin điều tra (`[SRC-2]`)."
        ),
    },
    "value": {
        "analysis": (
            "### 🔵 2. Value Investor (Dài hạn — Giá trị Nội tại & Biên An toàn)\n"
            "- **Hào kinh tế (Economic Moat) rất rộng:** Hệ sinh thái khép kín gồm GPU + kết nối NVLink + phần mềm CUDA tạo chi phí chuyển đổi cực cao, bảo vệ biên lợi nhuận dài hạn `[SRC-5]`.\n"
            "- **Định giá neo ở mức cao:** Với vốn hóa **$3.39T**, P/E trượt **54.2x** và P/E dự phóng **34.8x**, thị giá đã phản ánh phần lớn kỳ vọng tăng trưởng 2–3 năm tới `[FIN-1]` `[FIN-2]` `[FIN-3]` `[FIN-4]`.\n"
            "- **Biên an toàn (Margin of Safety) hạn chế:** Cổ phiếu đang ở vùng **định giá hợp lý (Fairly Priced)** cho một doanh nghiệp đầu ngành, chưa có chiết khấu đủ sâu để mua giá trị."
        ),
        "evidence_used": (
            "**📌 Căn cứ dữ liệu Value Investor đã chọn từ kho dùng chung:**\n"
            "- `[SRC-5]` Báo cáo Form 10-Q về rào cản chuyển đổi của CUDA & NVLink *(khẳng định lợi thế cạnh tranh bền vững)*.\n"
            "- `[FIN-2]`, `[FIN-3]`, `[FIN-4]` Vốn hóa $3.39T, Trailing P/E 54.2x, Forward P/E 34.8x *(đo lường mức độ đắt/rẻ so với EPS $2.55 `[FIN-5]`)*."
        ),
    },
    "devil": {
        "analysis": (
            "### 🔴 3. Devil’s Advocate (Phản biện & Stress-Test Rủi ro)\n"
            "- **Rủi ro pháp lý & Chống độc quyền:** Các cuộc điều tra của DOJ (Mỹ) và EU nhắm vào hệ sinh thái CUDA có thể làm suy yếu khả năng ràng buộc khách hàng `[SRC-2]`.\n"
            "- **Rủi ro vĩ mô & Tính bền vững CapEx:** Nếu các Hyperscalers cắt giảm chi tiêu hạ tầng AI do tỷ suất hoàn vốn (ROI) chưa tương xứng, tăng trưởng của NVDA sẽ giảm tốc mạnh `[SRC-3]`.\n"
            "- **Áp lực chuỗi cung ứng & Cạnh tranh dài hạn:** Nút thắt đóng gói CoWoS tại TSMC, xu hướng Big Tech tự làm chip **ASIC**, và sự dịch chuyển từ Training sang **Inference** đe dọa thị phần độc tôn `[SRC-3]` `[SRC-4]`."
        ),
        "evidence_used": (
            "**📌 Căn cứ dữ liệu Devil’s Advocate đã chọn từ kho dùng chung:**\n"
            "- `[SRC-2]` Tin điều tra chống độc quyền từ WSJ/Bloomberg *(truy xuất từ Tầng 2 với từ khóa `investigation`)*.\n"
            "- `[SRC-3]` Báo cáo chuỗi cung ứng CoWoS, rủi ro CapEx Hyperscalers & chip ASIC *(truy xuất từ Tầng 2 với từ khóa `concerns`, `supply`)*.\n"
            "- `[SRC-4]` Tranh luận về sự dịch chuyển sang tác vụ Inference *(khai thác góc nhìn trái chiều trong Tầng 3)*."
        ),
    },
    "lead": (
        "### 🟡 4. Lead Analyst — Kết luận Đồng thuận Cuối cùng\n"
        "**Xếp hạng Đồng thuận (Consensus Rating): `NEUTRAL WITH CAUTION` (Trung lập nhưng Thận trọng)**\n\n"
        "- **Đối chiếu lập luận từ 3 báo cáo độc lập:**\n"
        "  1. **Market Investor** chỉ ra động lượng doanh thu +94% và sức hút từ dòng chip Blackwell.\n"
        "  2. **Value Investor** xác nhận hào kinh tế CUDA vững chắc nhưng cảnh báo P/E 54.2x khiến **biên an toàn rất mỏng**.\n"
        "  3. **Devil’s Advocate** bổ sung các rủi ro trọng yếu về điều tra chống độc quyền, nút thắt CoWoS và nguy cơ giảm tốc CapEx từ nhóm Hyperscalers.\n"
        "- **Quyết định hành động:** Sự kết hợp giữa định giá cao (thiếu biên an toàn) và rủi ro pháp lý/vĩ mô gia tăng khiến tỷ lệ lợi nhuận/rủi ro ngắn hạn chưa hấp dẫn. Khuyến nghị **quan sát thận trọng**, chờ điểm vào lệnh có chiết khấu tốt hơn."
    ),
}

# ==============================================================================
# 4. HÀM THU THẬP DỮ LIỆU ĐA TẦNG & DÒ MODEL GEMINI KHẢ DỤNG
# ==============================================================================
def discover_gemini_models(api_key: str) -> List[str]:
    """Hỏi trực tiếp Google API để lấy danh sách model đang kích hoạt trên API Key."""
    clean_key = api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API Key bị từ chối (HTTP {http_err.code}): {err_body}")
    except Exception as e:
        raise RuntimeError(f"Không thể kết nối tới Google API: {e}")

    valid_models: List[str] = []
    for m in data.get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        short_name = m.get("name", "").replace("models/", "")
        lower_n = short_name.lower()
        if any(skip in lower_n for skip in ["embedding", "imagen", "veo", "aqa", "tts", "robotics", "computer-use"]):
            continue
        valid_models.append(short_name)

    if not valid_models:
        raise RuntimeError("API Key hợp lệ nhưng không có model generateContent nào được bật.")

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


def _fetch_yahoo_chart_price(ticker: str) -> Optional[float]:
    """Lấy giá trực tiếp từ endpoint v8/finance/chart (không cần cookie/crumb của .info)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?interval=1d&range=5d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        result = payload.get("chart", {}).get("result", [])
        if result:
            meta = result[0].get("meta", {})
            p = meta.get("regularMarketPrice") or meta.get("previousClose")
            if isinstance(p, (int, float)) and p > 0:
                return float(p)
    except Exception:
        pass
    return None


def _fetch_quantitative_metrics(ticker: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Lấy 6 chỉ số tài chính theo cơ chế 4 lớp chống chặn IP Cloud:
    1) yf.Ticker(ticker).info (đầy đủ nhất)
    2) yf.Ticker(ticker).fast_info + quarterly_income_stmt (khi .info bị Yahoo chặn rate-limit)
    3) Yahoo v8/finance/chart REST endpoint
    4) Bảng dự phòng định lượng cho các mã phổ biến để tránh N/A trên Cloud.
    """
    price: Optional[float] = None
    mcap: Optional[float] = None
    pe: Optional[float] = None
    fpe: Optional[float] = None
    eps: Optional[float] = None
    rev: Optional[float] = None
    source_note = f"Yahoo Finance API ({ticker})"

    # Lớp 1 & 2: yfinance (.info -> .fast_info -> .quarterly_income_stmt)
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)

        try:
            info = tk.info or {}
        except Exception:
            info = {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        mcap = info.get("marketCap")
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        eps = info.get("trailingEps")
        rev = info.get("revenueGrowth")

        if price is None or mcap is None:
            try:
                fi = tk.fast_info
                if price is None:
                    price = getattr(fi, "last_price", None) or fi.get("lastPrice")
                if mcap is None:
                    mcap = getattr(fi, "market_cap", None) or fi.get("marketCap")
                source_note = f"Yahoo Finance fast_info + Financials ({ticker})"
            except Exception:
                pass

        if eps is None or rev is None:
            try:
                q_inc = tk.quarterly_income_stmt
                if q_inc is not None and not q_inc.empty:
                    if eps is None and "Diluted EPS" in q_inc.index:
                        eps_series = q_inc.loc["Diluted EPS"].dropna()
                        if len(eps_series) >= 4:
                            eps = float(eps_series.iloc[:4].sum())
                    if rev is None and "Total Revenue" in q_inc.index:
                        rev_series = q_inc.loc["Total Revenue"].dropna()
                        if len(rev_series) >= 5 and rev_series.iloc[4] != 0:
                            rev = float((rev_series.iloc[0] - rev_series.iloc[4]) / abs(rev_series.iloc[4]))
            except Exception:
                pass
    except Exception:
        pass

    # Lớp 3: Endpoint chart trực tiếp nếu vẫn thiếu giá
    if price is None:
        chart_p = _fetch_yahoo_chart_price(ticker)
        if chart_p is not None:
            price = chart_p
            source_note = f"Yahoo Finance Chart API ({ticker})"

    # Lớp 4: Bù đắp các trường bị Yahoo Cloud IP block từ dữ liệu chuẩn
    fb = FALLBACK_QUOTES.get(ticker.upper())
    used_fallback = False
    if fb:
        if price is None:
            price = fb["price"]
            used_fallback = True
        if mcap is None:
            mcap = fb["mcap"]
            used_fallback = True
        if eps is None:
            eps = fb["eps"]
            used_fallback = True
        if pe is None:
            pe = (price / eps) if (price and eps and eps > 0) else fb["pe"]
            used_fallback = True
        if fpe is None:
            fpe = fb["fpe"]
            used_fallback = True
        if rev is None:
            rev = fb["rev"]
            used_fallback = True

    if pe is None and isinstance(price, (int, float)) and isinstance(eps, (int, float)) and eps > 0:
        pe = price / eps

    if used_fallback:
        source_note = f"Yahoo Finance + Reference Snapshot ({ticker})"

    def _fmt_mcap(val: Optional[float]) -> str:
        if not isinstance(val, (int, float)) or val <= 0:
            return "N/A"
        if val >= 1e12:
            return f"${val / 1e12:.2f}T"
        return f"${val / 1e9:,.2f}B"

    raw_pairs = [
        ("Giá hiện tại", f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"),
        ("Vốn hóa thị trường", _fmt_mcap(mcap)),
        ("Trailing P/E", f"{pe:.2f}x" if isinstance(pe, (int, float)) else "N/A"),
        ("Forward P/E", f"{fpe:.2f}x" if isinstance(fpe, (int, float)) else "N/A"),
        ("Diluted EPS", f"${eps:.2f}" if isinstance(eps, (int, float)) else "N/A"),
        ("Tăng trưởng DT (YoY)", f"+{rev * 100:.1f}%" if isinstance(rev, (int, float)) and rev >= 0 else (f"{rev * 100:.1f}%" if isinstance(rev, (int, float)) else "N/A")),
    ]

    metrics_list = [
        {"id": f"FIN-{idx}", "label": lbl, "value": val, "source": source_note}
        for idx, (lbl, val) in enumerate(raw_pairs, start=1)
    ]
    return metrics_list, source_note


def _search_google_news_rss(query: str, max_items: int = 2) -> List[Dict[str, str]]:
    """Tìm kiếm tin tức qua Google News RSS (không bị chặn rate-limit trên Cloud)."""
    encoded_q = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
    items: List[Dict[str, str]] = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()[:16]
            source_el = item.find("source")
            publisher = source_el.text.strip() if (source_el is not None and source_el.text) else "Google News"
            items.append({
                "publisher": f"{publisher} ({pub_date})" if pub_date else publisher,
                "url": link,
                "snippet": title,
            })
    except Exception:
        pass
    return items


def fetch_financial_and_rag(company: str, ticker: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Bước 1: Thu thập đầy đủ 2 nhóm dữ liệu gắn mã định danh [FIN-x] và [SRC-x]:
    - Nhóm 1: Chỉ số định lượng ([FIN-1]..[FIN-6]) qua cơ chế 4 lớp chống chặn IP Cloud
    - Nhóm 2: Tin tức & báo cáo 4 tầng ([SRC-1]..[SRC-n]) kèm nguồn, URL và đoạn trích.
    """
    metrics_list, _ = _fetch_quantitative_metrics(ticker)

    tier_configs = [
        (
            "Tầng 1: Tin chính thống & Kết quả kinh doanh",
            False,
            f"{company} {ticker} stock earnings revenue growth",
        ),
        (
            "Tầng 2: Tin tiêu cực & Rủi ro [Từ khóa: issues, concerns, investigation]",
            True,
            f"{company} {ticker} issues OR concerns OR investigation OR lawsuit",
        ),
        (
            "Tầng 2: Rủi ro cạnh tranh & Chuỗi cung ứng [Từ khóa: risks, short interest]",
            True,
            f"{company} {ticker} risks OR competition OR supply chain",
        ),
        (
            "Tầng 3: Tâm lý nhà đầu tư & Định giá (Seeking Alpha / Reddit)",
            False,
            f"{company} {ticker} stock valuation analysis reddit seeking alpha",
        ),
        (
            "Tầng 4: Lợi thế cạnh tranh & Báo cáo tài chính (Moat / 10-Q)",
            False,
            f"{company} {ticker} competitive advantage economic moat 10-Q",
        ),
    ]

    rag_sources: List[Dict[str, Any]] = []
    src_counter = 1

    ddgs_available = False
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for tier_label, is_crit, q in tier_configs:
                hits = list(ddgs.text(q, max_results=2))
                for h in hits:
                    ddgs_available = True
                    rag_sources.append({
                        "id": f"SRC-{src_counter}",
                        "tier": tier_label,
                        "is_critical": is_crit,
                        "publisher": h.get("title", "Web Source")[:85],
                        "query": q,
                        "url": h.get("href", ""),
                        "snippet": h.get("body", ""),
                    })
                    src_counter += 1
    except Exception:
        ddgs_available = False

    if not ddgs_available or len(rag_sources) < 3:
        for tier_label, is_crit, q in tier_configs:
            rss_hits = _search_google_news_rss(q, max_items=2)
            for h in rss_hits:
                rag_sources.append({
                    "id": f"SRC-{src_counter}",
                    "tier": tier_label,
                    "is_critical": is_crit,
                    "publisher": h["publisher"],
                    "query": q,
                    "url": h["url"],
                    "snippet": h["snippet"],
                })
                src_counter += 1

    if not rag_sources:
        rag_sources.append({
            "id": "SRC-1",
            "tier": "Tầng 1: Dữ liệu tổng hợp",
            "is_critical": False,
            "publisher": "Yahoo Finance Summary",
            "query": f"{company} ({ticker})",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "snippet": "Không lấy được tin tức trực tuyến; các tác tử phân tích dựa trên bộ chỉ số tài chính định lượng [FIN-1..FIN-6].",
        })

    return metrics_list, rag_sources


def call_gemini_rest(api_key: str, model_candidates: List[str], system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """Gọi thẳng Gemini REST API theo danh sách model thực tế của API Key."""
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
                    last_error = f"Model {model_name} không trả về nội dung."
                    continue
                return candidates[0]["content"]["parts"][0]["text"], model_name
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore")
            last_error = f"[{model_name}] HTTP {http_err.code}: {err_body}"
            if http_err.code in (404, 429, 503):
                continue
            raise RuntimeError(last_error)
        except Exception as exc:
            last_error = f"[{model_name}] Lỗi kết nối: {exc}"
            continue

    raise RuntimeError(f"Đã thử các model {model_candidates[:4]} nhưng đều thất bại. Chi tiết: {last_error}")


def split_analysis_and_evidence(raw_text: str) -> Tuple[str, str]:
    """Tách phần luận điểm chính và phần 'Căn cứ dữ liệu đã chọn lọc' để hiển thị đóng khung rõ ràng."""
    marker = "📌 Căn cứ dữ liệu đã chọn lọc:"
    if marker in raw_text:
        parts = raw_text.split(marker, 1)
        return parts[0].strip(), f"**{marker}**\n{parts[1].strip()}"
    return raw_text.strip(), ""


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


def render_shared_data_pool(container, company: str, ticker: str, metrics: List[Dict[str, str]], rag_sources: List[Dict[str, Any]], model_list: List[str] = None):
    """Hiển thị công khai toàn bộ Kho Dữ liệu Dùng chung ([FIN-x] và [SRC-x]) trên màn hình."""
    with container.container():
        st.markdown(f"### 🗄️ Kho Dữ liệu Đầu vào Dùng chung cho **{company} ({ticker})**")
        st.caption(
            "Cả 3 tác tử đầu tiên (Market Investor, Value Investor, Devil’s Advocate) đều đọc chung tập dữ liệu bên dưới, "
            "nhưng mỗi tác tử chỉ trích xuất những bằng chứng phù hợp với vai trò của mình."
        )
        fin_src = metrics[0]["source"] if metrics else "Yahoo Finance"
        info_line = f"📊 Nguồn định lượng: `{fin_src}`"
        if model_list:
            info_line += f" &nbsp;|&nbsp; 🤖 Model khả dụng: `{', '.join(model_list[:5])}`"
        st.caption(info_line)

        m_cols = st.columns(len(metrics))
        for idx, m in enumerate(metrics):
            m_cols[idx].metric(label=f"[{m['id']}] {m['label']}", value=m["value"])

        with st.expander(f"📚 Chi tiết {len(rag_sources)} Nguồn Tin tức & Báo cáo RAG ([SRC-1] → [SRC-{len(rag_sources)}]) — Dữ liệu gốc nuôi 3 Persona", expanded=True):
            for src in rag_sources:
                card_cls = "source-card source-critical" if src.get("is_critical") else "source-card source-normal"
                crit_badge = "<span class='badge-crit'>⚠️ NGUỒN PHẢN BIỆN (TẦNG 2)</span>" if src.get("is_critical") else ""
                link_html = f" &nbsp;|&nbsp; <a href='{src['url']}' target='_blank'>🔗 Mở nguồn gốc</a>" if src.get("url") else ""
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <span class="badge-src">{src['id']}</span> <b>{src['tier']}</b> {crit_badge}<br/>
                        <small style="color:#475569;"><b>Nguồn:</b> {src['publisher']} &nbsp;|&nbsp; <b>Truy vấn:</b> <code>{src['query']}</code>{link_html}</small>
                        <p style="margin: 0.35rem 0 0 0;">{src['snippet']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================
st.markdown(
    """
    <div class="top-banner">
        <h2>🧠 Demo Hệ thống Đa Tác tử & Phản biện (4-Persona Reasoning)</h2>
        <p>Minh bạch hóa toàn bộ nguồn dữ liệu đầu vào ([FIN-x], [SRC-x]) và truy vết chính xác căn cứ mà từng Persona sử dụng để ra quyết định.</p>
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
    company_input = st.text_input("Tên công ty:", value="NVIDIA", placeholder="VD: NVIDIA, Apple, Tesla...")
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
        help="Để mặc định để hệ thống tự chọn model đang bật trên API Key của bạn.",
    )

b_col1, b_col2 = st.columns([1.4, 1])
with b_col1:
    run_live_btn = st.button(
        "🚀 CHẠY PHÂN TÍCH THỰC TẾ (LIVE 4 PERSONAS + TRUY VẾT NGUỒN)",
        type="primary",
        use_container_width=True,
    )
with b_col2:
    run_demo_btn = st.button(
        "▶️ CHẠY MẪU NVDA (ĐẦY ĐỦ NGUỒN DỮ LIỆU & TRÍCH DẪN)",
        use_container_width=True,
    )

st.markdown("---")

tracker_placeholder = st.empty()
data_pool_placeholder = st.empty()

st.markdown("### 🧩 Kết quả Phân tích & Căn cứ Ra quyết định của Từng Persona")
col_p1, col_p2, col_p3 = st.columns(3)
card1_placeholder = col_p1.empty()
card2_placeholder = col_p2.empty()
card3_placeholder = col_p3.empty()
lead_placeholder = st.empty()

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ==============================================================================
# 6. XỬ LÝ KHI BẤM NÚT CHẠY
# ==============================================================================
if run_live_btn or run_demo_btn:
    is_live = run_live_btn

    if is_live and not api_key_input.strip():
        st.error("⚠️ Bạn chưa nhập **Google API Key**. Hãy dán khóa vào ô phía trên hoặc bấm nút **CHẠY MẪU NVDA** bên cạnh.")
    else:
        steps = [
            {"title": "Bước 1/5: Thu thập Kho Dữ liệu Dùng chung ([FIN-1..6] Định lượng + [SRC-1..n] Tin tức 4 tầng)", "desc": "Lấy chỉ số tài chính Yahoo Finance và truy vấn tin tức tích cực + tin tiêu cực (issues, concerns, investigation).", "state": "wait"},
            {"title": "Bước 2/5: Persona 1 — Market Investor (generate_market_outlook)", "desc": "Trích xuất tín hiệu động lượng & tâm lý từ kho dữ liệu dùng chung [Ngữ cảnh cô lập #1].", "state": "wait"},
            {"title": "Bước 3/5: Persona 2 — Value Investor (generate_value_analysis)", "desc": "Trích xuất chỉ số định giá & lợi thế cạnh tranh (Moat) từ kho dữ liệu dùng chung [Ngữ cảnh cô lập #2].", "state": "wait"},
            {"title": "Bước 4/5: Persona 3 — Devil’s Advocate (generate_devils_advocate_view)", "desc": "Trích xuất dữ liệu tiêu cực Tầng 2 để phản biện & stress-test rủi ro [Ngữ cảnh cô lập #3].", "state": "wait"},
            {"title": "Bước 5/5: Persona 4 — Lead Analyst (generate_final_summary)", "desc": "Tổng hợp độc lập 3 bản báo cáo trên (không đọc dữ liệu thô) và chốt xếp hạng cuối cùng.", "state": "wait"},
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
                metrics_list, rag_sources = fetch_financial_and_rag(cur_company, cur_ticker)
                steps[0]["desc"] = f"Đã nạp {len(metrics_list)} chỉ số [FIN] và {len(rag_sources)} nguồn tin [SRC] (Model: {model_candidates[0]})."
            else:
                time.sleep(0.4)
                cur_company = BENCHMARK_NVDA["company"]
                cur_ticker = BENCHMARK_NVDA["ticker"]
                metrics_list = BENCHMARK_NVDA["metrics"]
                rag_sources = BENCHMARK_NVDA["rag_sources"]

            steps[0]["state"] = "done"
            steps[0]["time"] = f"{time.perf_counter() - t0:.2f}s"
            render_pipeline_tracker(tracker_placeholder, steps)

            render_shared_data_pool(
                data_pool_placeholder,
                cur_company,
                cur_ticker,
                metrics_list,
                rag_sources,
                model_candidates if is_live else None,
            )

            fin_prompt_block = "\n".join(
                f"- [{m['id']}] {m['label']}: {m['value']} (Nguồn: {m['source']})" for m in metrics_list
            )
            rag_prompt_block = "\n".join(
                f"- [{s['id']}] ({s['tier']} | {s['publisher']}): {s['snippet']}" for s in rag_sources
            )
            shared_human_prompt = (
                f"Công ty: {cur_company} ({cur_ticker})\n\n"
                f"=== DỮ LIỆU ĐỊNH LƯỢNG ===\n{fin_prompt_block}\n\n"
                f"=== KHO TÀI LIỆU & TIN TỨC RAG ===\n{rag_prompt_block}"
            )

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
                m_body, m_ev = split_analysis_and_evidence(raw_m)
                market_analysis = f"### 🟢 1. Market Investor (Ngắn hạn)\n{m_body}"
                market_evidence = m_ev
            else:
                time.sleep(0.45)
                market_analysis = BENCHMARK_NVDA["market"]["analysis"]
                market_evidence = BENCHMARK_NVDA["market"]["evidence_used"]

            steps[1]["state"] = "done"
            steps[1]["time"] = f"{time.perf_counter() - t1:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            card1_placeholder.markdown(
                f"<div class='persona-card card-market'><span class='badge'>Ngữ cảnh cô lập #1</span>\n\n{market_analysis}"
                f"<div class='evidence-box'>{market_evidence}</div></div>",
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
                v_body, v_ev = split_analysis_and_evidence(raw_v)
                value_analysis = f"### 🔵 2. Value Investor (Dài hạn)\n{v_body}"
                value_evidence = v_ev
            else:
                time.sleep(0.45)
                value_analysis = BENCHMARK_NVDA["value"]["analysis"]
                value_evidence = BENCHMARK_NVDA["value"]["evidence_used"]

            steps[2]["state"] = "done"
            steps[2]["time"] = f"{time.perf_counter() - t2:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            card2_placeholder.markdown(
                f"<div class='persona-card card-value'><span class='badge'>Ngữ cảnh cô lập #2</span>\n\n{value_analysis}"
                f"<div class='evidence-box'>{value_evidence}</div></div>",
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
                d_body, d_ev = split_analysis_and_evidence(raw_d)
                devil_analysis = f"### 🔴 3. Devil’s Advocate (Phản biện)\n{d_body}"
                devil_evidence = d_ev
            else:
                time.sleep(0.45)
                devil_analysis = BENCHMARK_NVDA["devil"]["analysis"]
                devil_evidence = BENCHMARK_NVDA["devil"]["evidence_used"]

            steps[3]["state"] = "done"
            steps[3]["time"] = f"{time.perf_counter() - t3:.2f}s | {used_model_name}"
            render_pipeline_tracker(tracker_placeholder, steps)
            card3_placeholder.markdown(
                f"<div class='persona-card card-devil'><span class='badge'>Ngữ cảnh cô lập #3</span>\n\n{devil_analysis}"
                f"<div class='evidence-box-devil'>{devil_evidence}</div></div>",
                unsafe_allow_html=True,
            )

            # ---------------- BƯỚC 5: LEAD ANALYST ----------------
            steps[4]["state"] = "running"
            render_pipeline_tracker(tracker_placeholder, steps)
            t4 = time.perf_counter()

            if is_live:
                lead_input = (
                    f"Công ty: {cur_company} ({cur_ticker})\n\n"
                    f"--- BÁO CÁO 1: MARKET INVESTOR ---\n{market_analysis}\n{market_evidence}\n\n"
                    f"--- BÁO CÁO 2: VALUE INVESTOR ---\n{value_analysis}\n{value_evidence}\n\n"
                    f"--- BÁO CÁO 3: DEVIL'S ADVOCATE ---\n{devil_analysis}\n{devil_evidence}"
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
                f"<div class='card-lead'><span class='badge'>🚫 KHÔNG đọc dữ liệu thô — Chỉ tổng hợp từ 3 báo cáo độc lập</span>\n\n{lead_out}</div>",
                unsafe_allow_html=True,
            )

            st.session_state["last_result"] = {
                "steps": steps,
                "company": cur_company,
                "ticker": cur_ticker,
                "metrics": metrics_list,
                "rag_sources": rag_sources,
                "model_candidates": model_candidates if is_live else None,
                "market_analysis": market_analysis,
                "market_evidence": market_evidence,
                "value_analysis": value_analysis,
                "value_evidence": value_evidence,
                "devil_analysis": devil_analysis,
                "devil_evidence": devil_evidence,
                "lead": lead_out,
            }
            st.success(f"✅ Hoàn tất phân tích & truy vết nguồn dữ liệu cho {cur_company} ({cur_ticker}) bằng `{used_model_name}`!")

        except Exception as exc:
            st.error(f"❌ Quá trình chạy gặp lỗi: {exc}")

elif st.session_state["last_result"] is not None:
    saved = st.session_state["last_result"]
    render_pipeline_tracker(tracker_placeholder, saved["steps"])
    render_shared_data_pool(
        data_pool_placeholder,
        saved["company"],
        saved["ticker"],
        saved["metrics"],
        saved["rag_sources"],
        saved.get("model_candidates"),
    )
    card1_placeholder.markdown(
        f"<div class='persona-card card-market'><span class='badge'>Ngữ cảnh cô lập #1</span>\n\n{saved['market_analysis']}"
        f"<div class='evidence-box'>{saved['market_evidence']}</div></div>",
        unsafe_allow_html=True,
    )
    card2_placeholder.markdown(
        f"<div class='persona-card card-value'><span class='badge'>Ngữ cảnh cô lập #2</span>\n\n{saved['value_analysis']}"
        f"<div class='evidence-box'>{saved['value_evidence']}</div></div>",
        unsafe_allow_html=True,
    )
    card3_placeholder.markdown(
        f"<div class='persona-card card-devil'><span class='badge'>Ngữ cảnh cô lập #3</span>\n\n{saved['devil_analysis']}"
        f"<div class='evidence-box-devil'>{saved['devil_evidence']}</div></div>",
        unsafe_allow_html=True,
    )
    lead_placeholder.markdown(
        f"<div class='card-lead'><span class='badge'>🚫 KHÔNG đọc dữ liệu thô — Chỉ tổng hợp từ 3 báo cáo độc lập</span>\n\n{saved['lead']}</div>",
        unsafe_allow_html=True,
    )