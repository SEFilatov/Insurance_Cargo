
import os
import re
import json
import time
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI

# --- Config ---
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "").strip()
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "").strip()
YANDEX_BASE_URL = os.getenv("YANDEX_BASE_URL", "https://llm.api.cloud.yandex.net/v1").strip()
YANDEX_MODEL_URI = os.getenv("YANDEX_MODEL_URI", "").strip()  # if empty -> default below

TARIFF_URL = os.getenv("TARIFF_URL", "").strip()
TARIFF_BEARER = os.getenv("TARIFF_BEARER", "").strip()

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").strip()
SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC", "3600"))  # 1 hour

# Allowed values (align with your tariff engine config)
FRANCHISE_OPTIONS = [20000, 50000]
ROUTE_OPTIONS = ["РФ", "СНГ-РФ", "ВЕСЬ МИР-РФ"]
CONDITION_OPTIONS = ["NEW", "USED"]

# Cargo whitelist mapping (IDs are safe; rates are not here)
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

# --- Minimal session store (in-memory) ---
# For production: replace with Redis / DB.
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _now() -> int:
    return int(time.time())

def _get_session(session_id: str) -> Dict[str, Any]:
    s = _SESSIONS.get(session_id)
    if not s or s.get("expires_at", 0) < _now():
        s = {
            "stage": "collect",
            "data": {
                "cargo_class_id": None,
                "cargo_desc": None,
                "sum_insured_rub": None,
                "condition": None,
                "franchise_rub": None,
                "is_reefer": None,
                "route_zone": None,
            },
            "pending": {
                "cargo_confirm": None,  # {"proposed_id": "...", "proposed_name": "..."}
            },
            "expires_at": _now() + SESSION_TTL_SEC,
        }
        _SESSIONS[session_id] = s
    else:
        s["expires_at"] = _now() + SESSION_TTL_SEC
    return s

def _clean_sessions():
    # simple GC
    t = _now()
    dead = [k for k, v in _SESSIONS.items() if v.get("expires_at", 0) < t]
    for k in dead:
        _SESSIONS.pop(k, None)

# --- LLM client for cargo classification ---
if YANDEX_FOLDER_ID and YANDEX_API_KEY:
    _client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url=YANDEX_BASE_URL,
        project=YANDEX_FOLDER_ID,
    )
else:
    _client = None

def _default_model() -> str:
    if YANDEX_MODEL_URI:
        return YANDEX_MODEL_URI
    return f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite"

def llm_classify_cargo(desc: str) -> Tuple[Optional[str], str]:
    if not _client:
        return None, "LLM not configured"
    system = {
        "role": "system",
        "content": (
            "Ты классификатор грузов для страхования перевозок. "
            "Тебе дано описание груза и белый список допустимых классов. "
            "Верни строго JSON без пояснений вокруг. "
            "Если не уверен или груз не подходит — верни cargo_class_id=null.\n\n"
            f"Белый список (id -> name): {json.dumps(CARGO_CLASSES, ensure_ascii=False)}"
        ),
    }
    user = {
        "role": "user",
        "content": f"Описание груза: {desc}\nВерни JSON: {{\"cargo_class_id\": string|null, \"confidence\": 0..1, \"reason\": string}}",
    }
    resp = _client.chat.completions.create(
        model=_default_model(),
        messages=[system, user],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content or ""
    try:
        obj = json.loads(txt)
        cid = obj.get("cargo_class_id")
        reason = obj.get("reason", "")
        if cid in CARGO_CLASSES:
            return cid, reason
        return None, reason or "not in whitelist"
    except Exception:
        return None, "cannot parse classification"

def extract_fields(message: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ml = message.lower()

    # sum insured
    m = re.search(r"(\d[\d\s]{0,12})(?:\s*млн|\s*миллион)", ml)
    if m:
        num = int(re.sub(r"\s+", "", m.group(1)))
        out["sum_insured_rub"] = num * 1_000_000
    else:
        m2 = re.search(r"\b(\d[\d\s]{5,})\b", message)
        if m2:
            try:
                out["sum_insured_rub"] = int(re.sub(r"\s+", "", m2.group(1)))
            except Exception:
                pass

    # condition
    if "б/у" in ml or " бу " in ml or ml.startswith("бу") or "подерж" in ml:
        out["condition"] = "USED"
    elif "нов" in ml:
        out["condition"] = "NEW"

    # franchise (strict options)
    if ("франш" in ml or "фр" in ml) and "20" in ml:
        out["franchise_rub"] = 20000
    if ("франш" in ml or "фр" in ml) and "50" in ml:
        out["franchise_rub"] = 50000

    # reefer
    if "без реф" in ml or ("не" in ml and ("реф" in ml or "рефриж" in ml)):
        out["is_reefer"] = False
    elif "реф" in ml or "рефриж" in ml or "холод" in ml:
        out["is_reefer"] = True

    # route zone
    if "снг" in ml:
        out["route_zone"] = "СНГ-РФ"
    elif "весь мир" in ml:
        out["route_zone"] = "ВЕСЬ МИР-РФ"
    elif "рф" in ml or "росс" in ml:
        out["route_zone"] = "РФ"

    # cargo desc
    if any(x in ml for x in ["везу", "перевожу", "груз", "перевоз"]):
        out["cargo_desc"] = message.strip()
    return out

def call_tariff_engine(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TARIFF_URL:
        raise RuntimeError("TARIFF_URL is not set")
    headers = {"Content-Type": "application/json"}
    if TARIFF_BEARER:
        headers["Authorization"] = f"Bearer {TARIFF_BEARER}"
    r = requests.post(TARIFF_URL, headers=headers, json=payload, timeout=15)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Tariff engine error: {r.status_code} {r.text}")
    return r.json()

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    stage: str
    data: Dict[str, Any]

app = FastAPI(title="Insurance Dialog Service")

origins = ["*"] if ALLOW_ORIGINS == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    _clean_sessions()
    return {
        "status": "ok",
        "yandex_configured": bool(_client),
        "tariff_url_set": bool(TARIFF_URL),
        "sessions": len(_SESSIONS),
    }

def missing_fields(data: Dict[str, Any]) -> list:
    missing = []
    for k in ["sum_insured_rub", "cargo_class_id", "condition", "franchise_rub", "is_reefer", "route_zone"]:
        if data.get(k) is None:
            missing.append(k)
    return missing

def ask_next(data: Dict[str, Any]) -> str:
    if data.get("cargo_class_id") is None:
        return "Что вы перевозите (например: микроволновки, станок ЧПУ, шприцы)?"
    if data.get("sum_insured_rub") is None:
        return "Какая страховая сумма (в рублях) по этой перевозке?"
    if data.get("condition") is None:
        return "Груз новый или б/у? (NEW = новый, USED = б/у)"
    if data.get("franchise_rub") is None:
        return "Выберите франшизу: 20 000 ₽ или 50 000 ₽?"
    if data.get("is_reefer") is None:
        return "Нужен рефрижератор? (да/нет)"
    if data.get("route_zone") is None:
        return "Выберите зону маршрута: РФ / СНГ-РФ / ВЕСЬ МИР-РФ"
    return "Готово. Считаю стоимость…"

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    s = _get_session(req.session_id)
    data = s["data"]
    pending = s["pending"]
    txt = req.message.strip()
    tl = txt.lower()

    # handle cargo confirmation
    if pending.get("cargo_confirm"):
        if tl in ["да", "верно", "правильно", "ок", "yes", "y"]:
            data["cargo_class_id"] = pending["cargo_confirm"]["proposed_id"]
            pending["cargo_confirm"] = None
        elif tl in ["нет", "неверно", "no", "n"]:
            pending["cargo_confirm"] = None
            data["cargo_class_id"] = None
            data["cargo_desc"] = None
            return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                                reply="Ок, уточните, пожалуйста: что именно вы перевозите? (1–3 слова)")
        else:
            # treat as new cargo description
            data["cargo_desc"] = txt
            pending["cargo_confirm"] = None
            data["cargo_class_id"] = None

    # if already quoted, handle yes/no
    if s["stage"] == "quoted":
        if tl in ["да", "yes", "ок", "конечно"]:
            s["stage"] = "next_phase"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                                reply="Отлично. Следующий шаг — оформление (контактные данные и реквизиты). Подключим дальше.")
        if tl in ["нет", "no"]:
            s["stage"] = "collect"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                                reply="Ок. Хотите изменить параметры (сумма/франшиза/зона/реф) или рассчитать другой груз?")

    # extract fields
    extracted = extract_fields(txt)
    for k, v in extracted.items():
        if k == "cargo_desc":
            data["cargo_desc"] = v
        else:
            if data.get(k) is None and v is not None:
                data[k] = v

    # classify cargo if possible
    if data.get("cargo_class_id") is None and data.get("cargo_desc"):
        cid, _reason = llm_classify_cargo(data["cargo_desc"])
        if cid:
            pending["cargo_confirm"] = {"proposed_id": cid, "proposed_name": CARGO_CLASSES[cid]}
            return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                                reply=f"Похоже, ваш груз относится к категории: «{CARGO_CLASSES[cid]}». Верно? (да/нет)")
        return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                            reply="Не смог однозначно определить категорию груза. Опишите груз чуть точнее (1–3 слова).")

    # if ready -> call tariff engine
    if not missing_fields(data):
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
            s["stage"] = "quoted"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                                reply=f"Стоимость полиса: {result.get('premium_rub')} ₽. Согласны оформить? (да/нет)")
        if decision == "DECLINE":
            s["stage"] = "declined"
            reasons = ", ".join(result.get("reasons", [])) or "условия не подходят"
            return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                                reply=f"Онлайн-оформление недоступно: {reasons}. Хотите передать заявку менеджеру? (да/нет)")
        s["stage"] = "refer"
        reasons = ", ".join(result.get("reasons", [])) or "нужна проверка"
        return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data,
                            reply=f"Нужно согласование: {reasons}. Хотите передать заявку менеджеру? (да/нет)")

    # first-time intro
    intro = ""
    if s["stage"] == "collect" and all(v is None for v in data.values()):
        intro = (
            "Чтобы рассчитать стоимость, мне нужны: страховая сумма, груз, новый/б/у, франшиза (20k/50k), реф (да/нет), зона маршрута (РФ/СНГ-РФ/ВЕСЬ МИР-РФ).\n\n"
        )
    return ChatResponse(session_id=req.session_id, stage=s["stage"], data=data, reply=intro + ask_next(data))
