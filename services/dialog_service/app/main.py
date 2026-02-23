import os
import re
import json
import time
from typing import Any, Dict, Optional, Tuple, Literal

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

def _default_model() -> str:
    if YANDEX_MODEL_URI:
        return YANDEX_MODEL_URI
    return f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite"

LLMStatus = Literal["ok", "uncertain", "error"]

def llm_classify_cargo(
    desc: str,
    max_attempts: int = LLM_ATTEMPTS_PER_TRY,
    base_delay_sec: float = LLM_BASE_DELAY_SEC,
) -> Tuple[Optional[str], LLMStatus, str]:
    """
    Returns (cargo_class_id or None, status, reason).
    - ok: cargo_class_id valid
    - uncertain: model responded but cargo_class_id is null/invalid
    - error: LLM call failed (auth/quota/network/etc) or not configured
    Never raises.
    """
    if not _client:
        return None, "error", "LLM not configured"

    system = {
        "role": "system",
        "content": (
            "Ты классификатор грузов для страхования грузоперевозок. "
            "Тебе дано описание груза и белый список допустимых классов. "
            "Верни строго JSON без лишнего текста. "
            "Если не уверен или груз не подходит — верни cargo_class_id=null.\n\n"
            f"Белый список (id -> name): {json.dumps(CARGO_CLASSES, ensure_ascii=False)}"
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Описание груза: {desc}\n"
            "Верни JSON: {\"cargo_class_id\": string|null, \"confidence\": 0..1, \"reason\": string}"
        ),
    }

    last_reason = "unknown"
    last_uncertain_reason = "uncertain"

    for attempt in range(1, max_attempts + 1):
        try:
            resp = _client.chat.completions.create(
                model=_default_model(),
                messages=[system, user],
                temperature=0.0,
            )
            txt = resp.choices[0].message.content or ""
            obj = json.loads(txt)
            cid = obj.get("cargo_class_id")
            reason = obj.get("reason", "")

            if cid in CARGO_CLASSES:
                return cid, "ok", reason

            last_uncertain_reason = reason or "not in whitelist"
            last_reason = last_uncertain_reason

        except Exception as e:
            last_reason = f"LLM attempt {attempt} failed: {type(e).__name__}"

        if attempt < max_attempts:
            time.sleep(base_delay_sec * attempt)

    if "failed" in last_reason.lower() or "not configured" in last_reason.lower():
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
def call_tariff_engine(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TARIFF_URL:
        raise HTTPException(status_code=500, detail="TARIFF_URL is not set")

    headers = {"Content-Type": "application/json"}
    if TARIFF_BEARER:
        headers["Authorization"] = f"Bearer {TARIFF_BEARER}"

    try:
        r = requests.post(TARIFF_URL, headers=headers, json=payload, timeout=15)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tariff engine unreachable: {type(e).__name__}")

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Tariff engine error: {r.status_code} {r.text}")

    return r.json()

# =========================
# API models
# =========================
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    stage: str
    intent: Optional[str]
    data: Dict[str, Any]

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
        "model": _default_model() if YANDEX_FOLDER_ID else None,
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
    tl = text.lower().strip()

    # 1) Welcome -> Intent select
    if s["stage"] == "welcome":
        s["stage"] = "intent_select"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data, reply=WELCOME_TEXT)

    # 2) Intent selection
    if s["stage"] == "intent_select":
        if is_intent_consult(text):
            s["intent"] = "consult"
            s["stage"] = "consult"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply="Хорошо. Сформулируйте ваш вопрос по страхованию грузоперевозок — я отвечу."
            )
        if is_intent_buy(text):
            s["intent"] = "buy"
            s["stage"] = "quote_sum"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply="Отлично, оформим страховку. " + next_question("quote_sum")
            )
        return ChatResponse(
            session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
            reply="Пожалуйста, напишите: **консультация** или **оформить страховку**."
        )

    # 3) Consult mode (MVP)
    if s["stage"] == "consult":
        if is_intent_buy(text):
            s["intent"] = "buy"
            s["stage"] = "quote_sum"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply="Понял. Давайте оформим. " + next_question("quote_sum")
            )
        return ChatResponse(
            session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
            reply=(
                "Понял вопрос. Сейчас я могу подсказать общий порядок и требования.\n"
                "Если хотите — уточните: какой груз, какая сумма и зона перевозки.\n"
                "Если нужно сразу рассчитать стоимость — напишите: **оформить страховку**."
            )
        )

    # =========================
    # BUY FLOW
    # =========================

    # cargo_confirm
    if s["stage"] == "cargo_confirm" and pending.get("cargo_proposed"):
        yn = parse_yes_no(text)
        if yn is True:
            data["cargo_class_id"] = pending["cargo_proposed"]["id"]
            pending["cargo_proposed"] = None
            s["stage"] = "quote_condition"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply=next_question("quote_condition"))
        if yn is False:
            pending["cargo_proposed"] = None
            data["cargo_class_id"] = None
            data["cargo_desc"] = None
            s["stage"] = "quote_cargo"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Ок. Тогда уточните, пожалуйста, какой груз перевозите (1–2 слова)?")
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply="Подтвердите, пожалуйста: **да** или **нет**.")

    # cargo_retry
    if s["stage"] == "cargo_retry":
        pending["cargo_retry_count"] = int(pending.get("cargo_retry_count", 0)) + 1
        desc = data.get("cargo_desc") or ""

        cid, status, reason = llm_classify_cargo(desc)
        if status == "ok" and cid:
            pending["cargo_proposed"] = {"id": cid, "name": CARGO_CLASSES[cid]}
            s["stage"] = "cargo_confirm"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply=f"Похоже, ваш груз относится к категории: «{CARGO_CLASSES[cid]}». Верно? (да/нет)"
            )

        if status == "error" and pending["cargo_retry_count"] <= CARGO_RETRY_MAX:
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply=(
                    "Секунду, уточняю категорию груза… сервис классификации временно отвечает нестабильно.\n"
                    "Подождите 5–10 секунд и отправьте любое сообщение (например, «ок»), я попробую ещё раз."
                ),
            )

        s["stage"] = "cargo_choose"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply=manual_cargo_choice_text())

    # cargo_choose
    if s["stage"] == "cargo_choose":
        cid = parse_manual_cargo_choice(text)
        if cid:
            data["cargo_class_id"] = cid
            s["stage"] = "quote_condition"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply=f"Принял категорию: «{CARGO_CLASSES[cid]}».\n{next_question('quote_condition')}"
            )
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply="Пожалуйста, введите номер категории (1–16).")

    # quote_sum
    if s["stage"] == "quote_sum":
        val = parse_sum_rub(text)
        if val is None:
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Не понял сумму. Укажите, пожалуйста, например: 5 000 000 или 5 млн.")
        data["sum_insured_rub"] = val
        s["stage"] = "quote_cargo"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply="Спасибо. " + next_question("quote_cargo"))

    # quote_cargo
    if s["stage"] == "quote_cargo":
        data["cargo_desc"] = text
        pending["cargo_retry_count"] = 0

        cid, status, reason = llm_classify_cargo(text)

        if status == "ok" and cid:
            pending["cargo_proposed"] = {"id": cid, "name": CARGO_CLASSES[cid]}
            s["stage"] = "cargo_confirm"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply=f"Похоже, ваш груз относится к категории: «{CARGO_CLASSES[cid]}». Верно? (да/нет)"
            )

        if status == "error":
            s["stage"] = "cargo_retry"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply=(
                    "Секунду, уточняю категорию груза…\n"
                    "Подождите 5–10 секунд и отправьте любое сообщение (например, «ок»), я попробую ещё раз."
                ),
            )

        s["stage"] = "cargo_choose"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply=manual_cargo_choice_text())

    # quote_condition
    if s["stage"] == "quote_condition":
        cond = parse_condition(text)
        if cond is None:
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Укажите: NEW (новый) или USED (б/у).")
        data["condition"] = cond
        s["stage"] = "quote_franchise"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply=next_question("quote_franchise"))

    # quote_franchise
    if s["stage"] == "quote_franchise":
        fr = parse_franchise(text)
        if fr is None or fr not in FRANCHISE_OPTIONS:
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Выберите франшизу строго из списка: 20 000 ₽ или 50 000 ₽.")
        data["franchise_rub"] = fr
        s["stage"] = "quote_reefer"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply=next_question("quote_reefer"))

    # quote_reefer
    if s["stage"] == "quote_reefer":
        rr = parse_reefer(text)
        if rr is None:
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Нужен рефрижератор? Ответьте: да или нет.")
        data["is_reefer"] = rr
        s["stage"] = "quote_route"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply=next_question("quote_route"))

    # quote_route -> call tariff
    if s["stage"] == "quote_route":
        rz = parse_route_zone(text)
        if rz is None or rz not in ROUTE_OPTIONS:
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Выберите зону строго из списка: РФ / СНГ-РФ / ВЕСЬ МИР-РФ")
        data["route_zone"] = rz

        missing = quote_missing(data)
        if missing:
            s["stage"] = "quote_sum"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Не хватает данных для расчёта. " + next_question("quote_sum"))

        payload = {
            "cargo_class_id": data["cargo_class_id"],
            "sum_insured_rub": int(data["sum_insured_rub"]),
            "condition": data["condition"],
            "franchise_rub": int(data["franchise_rub"]),
            "is_reefer": bool(data["is_reefer"]),
            "route_zone": data["route_zone"],
        }

        result = call_tariff_engine(payload)
        decision = result.get("decision", "REFER")

        if decision == "AUTO_OK":
            premium = result.get("premium_rub")
            s["stage"] = "quoted"
            return ChatResponse(
                session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                reply=f"Стоимость страховки: {premium} ₽.\nСогласны оформить? (да/нет)"
            )

        reasons = ", ".join(result.get("reasons", [])) or "нужна проверка"
        s["stage"] = "refer"
        return ChatResponse(
            session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
            reply=f"Онлайн-оформление недоступно: {reasons}. Хотите передать заявку менеджеру? (да/нет)"
        )

    # quoted
    if s["stage"] == "quoted":
        yn = parse_yes_no(text)
        if yn is True:
            s["stage"] = "next_phase"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Отлично. Следующий шаг — ввод контактных данных и выпуск полиса. Эту фазу подключим дальше.")
        if yn is False:
            s["stage"] = "intent_select"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Ок. Хотите консультацию или рассчитать другую перевозку? (консультация / оформить страховку)")
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply="Ответьте, пожалуйста: да или нет.")

    # refer
    if s["stage"] == "refer":
        yn = parse_yes_no(text)
        if yn is True:
            s["stage"] = "handoff"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Принято. (MVP) Передача менеджеру будет подключена следующим шагом.")
        if yn is False:
            s["stage"] = "intent_select"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                                reply="Ок. Хотите консультацию или рассчитать другую перевозку? (консультация / оформить страховку)")
        return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                            reply="Ответьте, пожалуйста: да или нет.")

    # fallback
    s["stage"] = "intent_select"
    return ChatResponse(session_id=req.session_id, stage=s["stage"], intent=s["intent"], data=data,
                        reply="Давайте начнём: вам нужна консультация или оформить страховку?")