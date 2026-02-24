import os
import re
import json
import time
from typing import Any, Dict, Optional, Tuple, Literal, List

import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

# Для теста можно "*", для нормальной работы — конкретный домен GitHub Pages
origins = ["https://sergeybritok.github.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # или ["*"] для быстрого теста
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV / CONFIG
# =========================
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "").strip()
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "").strip()
YANDEX_BASE_URL = os.getenv("YANDEX_BASE_URL", "https://llm.api.cloud.yandex.net/v1").strip()
YANDEX_MODEL_URI = os.getenv("YANDEX_MODEL_URI", "").strip()

TARIFF_URL = os.getenv("TARIFF_URL", "").strip()
TARIFF_BEARER = os.getenv("TARIFF_BEARER", "").strip()

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").strip()
SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC", "3600"))  # 1 hour

# Retry controls for LLM cargo classification
CARGO_RETRY_MAX = int(os.getenv("CARGO_RETRY_MAX", "2"))      # additional user-triggered retries
LLM_ATTEMPTS_PER_TRY = int(os.getenv("LLM_ATTEMPTS_PER_TRY", "3"))
LLM_BASE_DELAY_SEC = float(os.getenv("LLM_BASE_DELAY_SEC", "0.6"))

# Debug size limits
DEBUG_MAX_TEXT = int(os.getenv("DEBUG_MAX_TEXT", "4000"))

# Strict options (align with tariff_service)
FRANCHISE_OPTIONS = [20000, 50000]
ROUTE_OPTIONS = ["РФ", "СНГ-РФ", "ВЕСЬ МИР-РФ"]
CONDITION_OPTIONS = ["NEW", "USED"]

# =========================
# Cargo whitelist (safe)
# =========================
CARGO_CLASSES: Dict[str, str] = {
    "CARGO001": "Автотранспортные средства и принадлежности к ним (в т.ч. пневматические шины)",
    "CARGO002": "Целлюлозно-бумажная продукция (бумага, картон, изделия из бумаги) и вторичное сырьё (макулатура)",
    "CARGO003": "Бытовая электротехника, электронная аппаратура и офисная (организационная) техника",
    "CARGO004": "Лекарственные средства (медикаменты), не требующие соблюдения температурного режима при перевозке и хранении",
    "CARGO005": "Автокомпоненты: запасные части, узлы, агрегаты и аксессуары к автотранспортным средствам",
    "CARGO006": "Фармацевтическая продукция (прочая/субстанции и т.п.), если принимается по условиям страхования",
    "CARGO007": "Строительные материалы, отделочные материалы и строительно-монтажный инструмент",
    "CARGO008": "Парфюмерно-косметическая продукция и товары бытовой химии",
    "CARGO009": "Мебель и предметы интерьера",
    "CARGO010": "Товары народного потребления (прочие), не поименованные отдельно в перечне допустимых грузов",
    "CARGO011": "Продовольственные товары (пищевые продукты)",
    "CARGO012": "Детские товары (в т.ч. игрушки) и товары для спорта и отдыха (спортивный инвентарь)",
    "CARGO013": "Оборудование промышленное/технологическое (в т.ч. станки, агрегаты, производственные линии)",
    "CARGO014": "Изделия медицинского назначения: медицинская техника, инструменты и медицинские изделия",
    "CARGO015": "Металлопрокат и изделия из металла (металлоизделия)",
    "CARGO016": "Химическая продукция неопасная (не классифицируемая как опасный груз/ADR)",
}
CARGO_ORDER = list(CARGO_CLASSES.keys())

# =========================
# LLM client (ONLY for cargo classification)
# =========================
_client: Optional[OpenAI] = None
if YANDEX_FOLDER_ID and YANDEX_API_KEY:
    _client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url=YANDEX_BASE_URL,
        project=YANDEX_FOLDER_ID,
    )

def _default_model() -> Optional[str]:
    if not YANDEX_FOLDER_ID:
        return None
    
    if YANDEX_MODEL_URI:
        return YANDEX_MODEL_URI
    return f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite"

LLMStatus = Literal["ok", "uncertain", "error"]

def _clip(s: str, n: int = DEBUG_MAX_TEXT) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n…(truncated, {len(s)} chars total)"

def llm_classify_cargo_with_trace(
    desc: str,
    debug_enabled: bool,
    debug_trace: Dict[str, Any],
    max_attempts: int = LLM_ATTEMPTS_PER_TRY,
    base_delay_sec: float = LLM_BASE_DELAY_SEC,
) -> Tuple[Optional[str], LLMStatus, str]:
    """
    Calls LLM and appends full request/response to debug_trace['llm_calls'] when debug_enabled.
    Returns (cargo_class_id or None, status, reason). Never raises.
    """
    if not _client:
        if debug_enabled:
            debug_trace["llm_calls"].append({
                "kind": "cargo_classification",
                "status": "error",
                "reason": "LLM not configured",
                "input": desc,
                "model": _default_model(),
            })
        return None, "error", "LLM not configured"

    system_msg = (
        "Ты классификатор грузов для страхования грузоперевозок. "
        "Тебе дано описание груза и белый список допустимых классов. "
        "Верни строго JSON без лишнего текста. "
        "Если не уверен или груз не подходит — верни cargo_class_id=null.\n\n"
        f"Белый список (id -> name): {json.dumps(CARGO_CLASSES, ensure_ascii=False)}"
    )
    user_msg = (
        f"Описание груза: {desc}\n"
        "Верни JSON: {\"cargo_class_id\": string|null, \"confidence\": 0..1, \"reason\": string}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    last_reason = "unknown"
    last_uncertain_reason = "uncertain"
    last_raw = ""
    model = _default_model()

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        err_name = None
        parsed_obj = None
        cid = None
        status: LLMStatus = "error"

        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            last_raw = resp.choices[0].message.content or ""
            # parse
            parsed_obj = json.loads(last_raw)
            cid = parsed_obj.get("cargo_class_id")
            reason = parsed_obj.get("reason", "")
            if cid in CARGO_CLASSES:
                status = "ok"
                last_reason = reason
            else:
                status = "uncertain"
                last_uncertain_reason = reason or "not in whitelist"
                last_reason = last_uncertain_reason
        except Exception as e:
            err_name = type(e).__name__
            last_reason = f"LLM attempt {attempt} failed: {err_name}"
            status = "error"

        elapsed_ms = int((time.time() - t0) * 1000)

        if debug_enabled:
            debug_trace["llm_calls"].append({
                "kind": "cargo_classification",
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
                "model": model,
                "request": {
                    "messages": [
                        {"role": "system", "content": _clip(system_msg)},
                        {"role": "user", "content": _clip(user_msg)},
                    ]
                },
                "response_raw": _clip(last_raw),
                "response_parsed": parsed_obj,
                "status": status,
                "cargo_class_id": cid if cid in CARGO_CLASSES else None,
                "cargo_class_name": CARGO_CLASSES.get(cid) if cid in CARGO_CLASSES else None,
                "reason": last_reason,
                "error": err_name,
            })

        if status == "ok" and cid in CARGO_CLASSES:
            return cid, "ok", last_reason

        # If uncertain, we can retry (sometimes output formatting is unstable)
        # If error, retry as well.
        if attempt < max_attempts:
            time.sleep(base_delay_sec * attempt)

    # Final decision after attempts
    # If last attempt was a real LLM error -> error, else uncertain
    if "failed" in last_reason.lower() or "not configured" in last_reason.lower() or "attempt" in last_reason.lower():
        return None, "error", last_reason
    return None, "uncertain", last_uncertain_reason or last_reason

# =========================
# Session store (in-memory MVP)
# =========================
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _now() -> int:
    return int(time.time())

def _clean_sessions():
    t = _now()
    dead = [k for k, v in _SESSIONS.items() if v.get("expires_at", 0) < t]
    for k in dead:
        _SESSIONS.pop(k, None)

def _new_session() -> Dict[str, Any]:
    return {
        "stage": "welcome",
        "intent": None,   # None | consult | buy
        "data": {
            "sum_insured_rub": None,
            "cargo_desc": None,
            "cargo_class_id": None,
            "condition": None,
            "franchise_rub": None,
            "is_reefer": None,
            "route_zone": None,
        },
        "pending": {
            "cargo_proposed": None,   # {"id": "...", "name": "..."}
            "cargo_retry_count": 0,
        },
        "expires_at": _now() + SESSION_TTL_SEC,
    }

def _get_session(session_id: str) -> Dict[str, Any]:
    s = _SESSIONS.get(session_id)
    if not s or s.get("expires_at", 0) < _now():
        s = _new_session()
        _SESSIONS[session_id] = s
    s["expires_at"] = _now() + SESSION_TTL_SEC
    return s

# =========================
# Parsing helpers
# =========================
def parse_sum_rub(text: str) -> Optional[int]:
    t = text.lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(млн|миллион|million)", t)
    if m:
        val = float(m.group(1))
        return int(val * 1_000_000)

    m2 = re.search(r"\b(\d[\d\s]{5,})\b", text)
    if m2:
        try:
            return int(re.sub(r"\s+", "", m2.group(1)))
        except Exception:
            return None
    return None

def parse_condition(text: str) -> Optional[str]:
    tl = text.lower().strip()
    if "б/у" in tl or "бу" in tl or "подерж" in tl:
        return "USED"
    if "нов" in tl:
        return "NEW"
    if tl in ["new", "used"]:
        return tl.upper()
    return None

def parse_franchise(text: str) -> Optional[int]:
    tl = text.lower().strip()
    if tl in ["20000", "20", "20к", "20 к", "20 тыс"]:
        return 20000
    if tl in ["50000", "50", "50к", "50 к", "50 тыс"]:
        return 50000
    if ("франш" in tl or "франшиза" in tl or "фр" in tl) and any(x in tl for x in ["20", "20000", "20к", "20 тыс"]):
        return 20000
    if ("франш" in tl or "франшиза" in tl or "фр" in tl) and any(x in tl for x in ["50", "50000", "50к", "50 тыс"]):
        return 50000
    return None

def parse_yes_no(text: str) -> Optional[bool]:
    tl = text.strip().lower()
    if tl in ["да", "yes", "y", "ок", "ага", "конечно", "верно", "правильно"]:
        return True
    if tl in ["нет", "no", "n", "неа", "неверно"]:
        return False
    return None

def parse_reefer(text: str) -> Optional[bool]:
    tl = text.lower().strip()
    yn = parse_yes_no(text)
    if yn is not None:
        return yn
    if "без реф" in tl or ("не" in tl and ("реф" in tl or "рефриж" in tl)):
        return False
    if "реф" in tl or "рефриж" in tl or "холод" in tl:
        return True
    return None

def parse_route_zone(text: str) -> Optional[str]:
    tl = text.lower().strip()
    if "снг" in tl:
        return "СНГ-РФ"
    if "весь мир" in tl:
        return "ВЕСЬ МИР-РФ"
    if tl in ["rf", "russia", "россия", "рф"]:
        return "РФ"
    if text.strip() in ROUTE_OPTIONS:
        return text.strip()
    return None

def parse_manual_cargo_choice(text: str) -> Optional[str]:
    tl = text.strip().upper()
    if tl in CARGO_CLASSES:
        return tl
    if re.fullmatch(r"\d{1,2}", text.strip()):
        n = int(text.strip())
        if 1 <= n <= len(CARGO_ORDER):
            return CARGO_ORDER[n - 1]
    return None

def manual_cargo_choice_text() -> str:
    lines = ["Не смог автоматически определить категорию. Выберите номер категории из списка:"]
    for i, cid in enumerate(CARGO_ORDER, start=1):
        lines.append(f"{i}) {CARGO_CLASSES[cid]}")
    lines.append("Напишите номер (1–16).")
    return "\n".join(lines)

# =========================
# Tariff engine call
# =========================
def call_tariff_engine(payload: Dict[str, Any], debug_enabled: bool, debug_trace: Dict[str, Any]) -> Dict[str, Any]:
    if not TARIFF_URL:
        raise HTTPException(status_code=500, detail="TARIFF_URL is not set")

    headers = {"Content-Type": "application/json"}
    if TARIFF_BEARER:
        headers["Authorization"] = f"Bearer {TARIFF_BEARER}"

    t0 = time.time()
    try:
        r = requests.post(TARIFF_URL, headers=headers, json=payload, timeout=15)
    except Exception as e:
        if debug_enabled:
            debug_trace["tariff_call"] = {
                "request": payload,
                "error": type(e).__name__,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
        raise HTTPException(status_code=502, detail=f"Tariff engine unreachable: {type(e).__name__}")

    elapsed_ms = int((time.time() - t0) * 1000)
    if r.status_code >= 400:
        if debug_enabled:
            debug_trace["tariff_call"] = {
                "request": payload,
                "status_code": r.status_code,
                "response_text": _clip(r.text),
                "elapsed_ms": elapsed_ms,
            }
        raise HTTPException(status_code=502, detail=f"Tariff engine error: {r.status_code} {r.text}")

    result = r.json()
    if debug_enabled:
        debug_trace["tariff_call"] = {
            "request": payload,
            "response": result,
            "elapsed_ms": elapsed_ms,
        }
    return result

# =========================
# API models
# =========================
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    debug: bool = False

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    stage: str
    intent: Optional[str]
    data: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None

# =========================
# FastAPI app + CORS
# =========================
app = FastAPI(title="Insurance Dialog Service")

origins = ["*"] if ALLOW_ORIGINS == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/chat")
def chat_options():
    return Response(status_code=200)

@app.get("/health")
def health():
    _clean_sessions()
    return {
        "status": "ok",
        "yandex_configured": bool(_client),
        "tariff_url_set": bool(TARIFF_URL),
        "sessions": len(_SESSIONS),
        "allow_origins": origins,
        "model": _default_model(),
    }

# =========================
# Dialog text & helpers
# =========================
WELCOME_TEXT = (
    "Здравствуйте! Я ассистент по страхованию грузоперевозок.\n"
    "Могу проконсультировать по продукту и за несколько минут рассчитать стоимость страховки по вашей перевозке.\n"
    "Я задам несколько вопросов и после этого рассчитаю стоимость через тарифный сервис.\n"
    "Что выбираете: **консультация** или **оформить страховку**?"
)

def is_intent_consult(text: str) -> bool:
    tl = text.lower()
    return "консул" in tl or "вопрос" in tl or "услов" in tl

def is_intent_buy(text: str) -> bool:
    tl = text.lower()
    return "оформ" in tl or "страхов" in tl or "полис" in tl or "рассч" in tl or "куп" in tl

def quote_missing(data: Dict[str, Any]) -> list:
    miss = []
    for k in ["sum_insured_rub", "cargo_class_id", "condition", "franchise_rub", "is_reefer", "route_zone"]:
        if data.get(k) is None:
            miss.append(k)
    return miss

def next_question(stage: str) -> str:
    if stage == "quote_sum":
        return "Какая страховая сумма по перевозке (в рублях)? Например: 5 000 000 или 5 млн."
    if stage == "quote_cargo":
        return "Какой груз перевозите? Напишите одним-двумя словами (например: молоко, микроволновки, шприцы, станок ЧПУ)."
    if stage == "quote_condition":
        return "Груз новый или б/у? (NEW = новый, USED = б/у)"
    if stage == "quote_franchise":
        return "Выберите франшизу: 20 000 ₽ или 50 000 ₽?"
    if stage == "quote_reefer":
        return "Нужен рефрижератор? (да/нет)"
    if stage == "quote_route":
        return "Выберите зону маршрута: РФ / СНГ-РФ / ВЕСЬ МИР-РФ"
    return "Уточните, пожалуйста."

# =========================
# Main handler
# =========================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    s = _get_session(req.session_id)
    data = s["data"]
    pending = s["pending"]

    text = req.message.strip()
    debug_enabled = bool(req.debug)

    # Per-turn debug trace (returned to widget only when debug=true)
    debug_trace: Dict[str, Any] = {
        "turn": {
            "session_id": req.session_id,
            "incoming_message": text,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "session_before": {
            "stage": s.get("stage"),
            "intent": s.get("intent"),
            "data": data.copy(),
            "pending": {
                "cargo_proposed": pending.get("cargo_proposed"),
                "cargo_retry_count": pending.get("cargo_retry_count"),
            },
        },
        "llm_calls": [],
        "tariff_call": None,
        "notes": [],
    }

    def respond(reply: str) -> ChatResponse:
        # attach session_after snapshot
        if debug_enabled:
            debug_trace["session_after"] = {
                "stage": s.get("stage"),
                "intent": s.get("intent"),
                "data": data.copy(),
                "pending": {
                    "cargo_proposed": pending.get("cargo_proposed"),
                    "cargo_retry_count": pending.get("cargo_retry_count"),
                },
            }
        return ChatResponse(
            session_id=req.session_id,
            reply=reply,
            stage=s.get("stage"),
            intent=s.get("intent"),
            data=data,
            debug=debug_trace if debug_enabled else None,
        )

    # 1) Welcome -> intent_select
    if s["stage"] == "welcome":
        s["stage"] = "intent_select"
        return respond(WELCOME_TEXT)

    # 2) Intent selection
    if s["stage"] == "intent_select":
        if is_intent_consult(text):
            s["intent"] = "consult"
            s["stage"] = "consult"
            return respond("Хорошо. Сформулируйте ваш вопрос по страхованию грузоперевозок — я отвечу.")
        if is_intent_buy(text):
            s["intent"] = "buy"
            s["stage"] = "quote_sum"
            return respond("Отлично, оформим страховку. " + next_question("quote_sum"))
        return respond("Пожалуйста, напишите: **консультация** или **оформить страховку**.")

    # 3) Consult mode (MVP placeholder; later RAG)
    if s["stage"] == "consult":
        if is_intent_buy(text):
            s["intent"] = "buy"
            s["stage"] = "quote_sum"
            return respond("Понял. Давайте оформим. " + next_question("quote_sum"))
        return respond(
            "Понял вопрос. Сейчас я могу подсказать общий порядок и требования.\n"
            "Если хотите — уточните: какой груз, какая сумма и зона перевозки.\n"
            "Если нужно сразу рассчитать стоимость — напишите: **оформить страховку**."
        )

    # =========================
    # BUY FLOW
    # =========================

    # cargo_confirm: confirm proposed cargo class
    if s["stage"] == "cargo_confirm" and pending.get("cargo_proposed"):
        yn = parse_yes_no(text)
        if yn is True:
            data["cargo_class_id"] = pending["cargo_proposed"]["id"]
            pending["cargo_proposed"] = None
            s["stage"] = "quote_condition"
            return respond(next_question("quote_condition"))
        if yn is False:
            pending["cargo_proposed"] = None
            data["cargo_class_id"] = None
            data["cargo_desc"] = None
            s["stage"] = "quote_cargo"
            return respond("Ок. Тогда уточните, пожалуйста, какой груз перевозите (1–2 слова)?")
        return respond("Подтвердите, пожалуйста: **да** или **нет**.")

    # cargo_retry: retry classification after user "waits"
    if s["stage"] == "cargo_retry":
        pending["cargo_retry_count"] = int(pending.get("cargo_retry_count", 0)) + 1
        desc = data.get("cargo_desc") or ""

        cid, status, reason = llm_classify_cargo_with_trace(desc, debug_enabled, debug_trace)
        if status == "ok" and cid:
            pending["cargo_proposed"] = {"id": cid, "name": CARGO_CLASSES[cid]}
            s["stage"] = "cargo_confirm"
            return respond(f"Похоже, ваш груз относится к категории: «{CARGO_CLASSES[cid]}». Верно? (да/нет)")

        if status == "error" and pending["cargo_retry_count"] <= CARGO_RETRY_MAX:
            return respond(
                "Секунду, уточняю категорию груза… сервис классификации временно отвечает нестабильно.\n"
                "Подождите 5–10 секунд и отправьте любое сообщение (например, «ок»), я попробую ещё раз."
            )

        s["stage"] = "cargo_choose"
        return respond(manual_cargo_choice_text())

    # cargo_choose: manual selection
    if s["stage"] == "cargo_choose":
        cid = parse_manual_cargo_choice(text)
        if cid:
            data["cargo_class_id"] = cid
            s["stage"] = "quote_condition"
            return respond(f"Принял категорию: «{CARGO_CLASSES[cid]}».\n{next_question('quote_condition')}")
        return respond("Пожалуйста, введите номер категории (1–16).")

    # quote_sum
    if s["stage"] == "quote_sum":
        val = parse_sum_rub(text)
        if val is None:
            return respond("Не понял сумму. Укажите, пожалуйста, например: 5 000 000 или 5 млн.")
        data["sum_insured_rub"] = val
        s["stage"] = "quote_cargo"
        return respond("Спасибо. " + next_question("quote_cargo"))

    # quote_cargo
    if s["stage"] == "quote_cargo":
        data["cargo_desc"] = text
        pending["cargo_retry_count"] = 0

        cid, status, reason = llm_classify_cargo_with_trace(text, debug_enabled, debug_trace)

        if status == "ok" and cid:
            pending["cargo_proposed"] = {"id": cid, "name": CARGO_CLASSES[cid]}
            s["stage"] = "cargo_confirm"
            return respond(f"Похоже, ваш груз относится к категории: «{CARGO_CLASSES[cid]}». Верно? (да/нет)")

        if status == "error":
            s["stage"] = "cargo_retry"
            return respond(
                "Секунду, уточняю категорию груза…\n"
                "Подождите 5–10 секунд и отправьте любое сообщение (например, «ок»), я попробую ещё раз."
            )

        s["stage"] = "cargo_choose"
        return respond(manual_cargo_choice_text())

    # quote_condition
    if s["stage"] == "quote_condition":
        cond = parse_condition(text)
        if cond is None:
            return respond("Укажите: NEW (новый) или USED (б/у).")
        data["condition"] = cond
        s["stage"] = "quote_franchise"
        return respond(next_question("quote_franchise"))

    # quote_franchise
    if s["stage"] == "quote_franchise":
        fr = parse_franchise(text)
        if fr is None or fr not in FRANCHISE_OPTIONS:
            return respond("Выберите франшизу строго из списка: 20 000 ₽ или 50 000 ₽.")
        data["franchise_rub"] = fr
        s["stage"] = "quote_reefer"
        return respond(next_question("quote_reefer"))

    # quote_reefer
    if s["stage"] == "quote_reefer":
        rr = parse_reefer(text)
        if rr is None:
            return respond("Нужен рефрижератор? Ответьте: да или нет.")
        data["is_reefer"] = rr
        s["stage"] = "quote_route"
        return respond(next_question("quote_route"))

    # quote_route -> call tariff
    if s["stage"] == "quote_route":
        rz = parse_route_zone(text)
        if rz is None or rz not in ROUTE_OPTIONS:
            return respond("Выберите зону строго из списка: РФ / СНГ-РФ / ВЕСЬ МИР-РФ")
        data["route_zone"] = rz

        missing = quote_missing(data)
        if missing:
            s["stage"] = "quote_sum"
            return respond("Не хватает данных для расчёта. " + next_question("quote_sum"))

        payload = {
            "cargo_class_id": data["cargo_class_id"],
            "sum_insured_rub": int(data["sum_insured_rub"]),
            "condition": data["condition"],
            "franchise_rub": int(data["franchise_rub"]),
            "is_reefer": bool(data["is_reefer"]),
            "route_zone": data["route_zone"],
        }

        result = call_tariff_engine(payload, debug_enabled, debug_trace)
        decision = result.get("decision", "REFER")

        if decision == "AUTO_OK":
            premium = result.get("premium_rub")
            s["stage"] = "quoted"
            return respond(f"Стоимость страховки: {premium} ₽.\nСогласны оформить? (да/нет)")

        reasons = ", ".join(result.get("reasons", [])) or "нужна проверка"
        s["stage"] = "refer"
        return respond(f"Онлайн-оформление недоступно: {reasons}. Хотите передать заявку менеджеру? (да/нет)")

    # quoted
    if s["stage"] == "quoted":
        yn = parse_yes_no(text)
        if yn is True:
            s["stage"] = "next_phase"
            return respond("Отлично. Следующий шаг — ввод контактных данных и выпуск полиса. Эту фазу подключим дальше.")
        if yn is False:
            s["stage"] = "intent_select"
            return respond("Ок. Хотите консультацию или рассчитать другую перевозку? (консультация / оформить страховку)")
        return respond("Ответьте, пожалуйста: да или нет.")

    # refer
    if s["stage"] == "refer":
        yn = parse_yes_no(text)
        if yn is True:
            s["stage"] = "handoff"
            return respond("Принято. (MVP) Передача менеджеру будет подключена следующим шагом.")
        if yn is False:
            s["stage"] = "intent_select"
            return respond("Ок. Хотите консультацию или рассчитать другую перевозку? (консультация / оформить страховку)")
        return respond("Ответьте, пожалуйста: да или нет.")

    # fallback
    s["stage"] = "intent_select"
    return respond("Давайте начнём: вам нужна консультация или оформить страховку?")