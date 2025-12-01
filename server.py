from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import requests

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "reviews.csv"
GENERATED_PATH = BASE_DIR / "generated_reviews.json"
CSV_FIELDS = [
    "review_id",
    "review_text",
    "student_name",
    "teacher_name",
    "course_name",
    "created_date",
    "rating",
    "sentiment",
    "word_count",
]

app = Flask(__name__)
load_dotenv()
_gigachat_token_cache: Dict[str, Any] = {
    "token": None,
    "expires_at": datetime.min,
}


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _gigachat_verify_tls() -> Any:
    ca_bundle = os.getenv("GIGACHAT_CA_BUNDLE")
    if ca_bundle:
        return ca_bundle
    return _env_flag("GIGACHAT_VERIFY_SSL", False)


def _sanitize_csv_lines(raw_lines: List[str]) -> List[str]:
    """Fix broken quotes in the CSV exported from Excel."""
    cleaned: List[str] = []
    for line in raw_lines:
        if not line.strip():
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        line = line.replace('""', '"')
        cleaned.append(line)
    return cleaned


def _parse_csv_lines(lines: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Try to parse CSV; if fields are missing (broken quoting), return None."""
    try:
        reader = csv.DictReader(lines)
        rows = list(reader)
    except Exception:
        return None

    if not rows:
        return []

    has_missing = any(None in row.values() or None in row.keys() for row in rows)
    return None if has_missing else rows


def _save_reviews(rows: List[Dict[str, Any]]) -> None:
    """Rewrite CSV with proper quoting in cp1251."""
    safe_rows: List[Dict[str, Any]] = []
    for row in rows:
        cleaned = {field: row.get(field, "") for field in CSV_FIELDS}
        safe_rows.append(cleaned)

    with CSV_PATH.open("w", encoding="cp1251", newline="", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(safe_rows)


def _normalize_student(name: str) -> str:
    return (name or "").replace('"', "").strip().lower()


def load_reviews() -> List[Dict[str, Any]]:
    if not CSV_PATH.exists():
        return []

    text = CSV_PATH.read_text(encoding="cp1251")
    lines = text.splitlines()
    if not lines:
        return []

    parsed = _parse_csv_lines(lines)
    if parsed is None:
        header, *raw_rows = lines
        parsed = _parse_csv_lines([header, *_sanitize_csv_lines(raw_rows)])
    if parsed is None:
        return []

    reader = parsed
    reviews: List[Dict[str, Any]] = []
    for row in reader:
        record: Dict[str, Any] = dict(row)
        try:
            record["rating"] = int(record.get("rating", "") or 0)
        except ValueError:
            record["rating"] = 0

        date_raw = record.get("created_date") or ""
        try:
            record["_created_dt"] = datetime.fromisoformat(date_raw)
        except Exception:
            record["_created_dt"] = datetime.min

        reviews.append(record)
    return reviews


def get_last_reviews(student_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    target = _normalize_student(student_name)
    reviews = load_reviews()
    filtered = [
        r
        for r in reviews
        if target and _normalize_student(r.get("student_name", "")) == target
    ]
    filtered.sort(key=lambda r: r["_created_dt"], reverse=True)
    result: List[Dict[str, Any]] = []
    for row in filtered[:limit]:
        row_copy = dict(row)
        row_copy.pop("_created_dt", None)
        result.append(row_copy)
    return result


def _cached_gigachat_token() -> Optional[str]:
    now = datetime.utcnow()
    token = _gigachat_token_cache.get("token")
    expires_at = _gigachat_token_cache.get("expires_at") or datetime.min
    return token if token and now + timedelta(seconds=90) < expires_at else None


def get_gigachat_token() -> Tuple[Optional[str], Optional[str]]:
    cached = _cached_gigachat_token()
    if cached:
        return cached, None

    auth_key = os.getenv("GIGACHAT_AUTH_KEY") or ""
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    if not auth_key:
        return None, "Не задан GIGACHAT_AUTH_KEY в .env"

    url = os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {auth_key}",
    }
    data = {"scope": scope}

    try:
        resp = requests.post(
            url,
            headers=headers,
            data=data,
            timeout=10,
            verify=_gigachat_verify_tls(),
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            return None, "ГигаЧат не вернул access_token"

        expires_at = datetime.utcnow() + timedelta(minutes=25)
        expires_raw = payload.get("expires_at") or payload.get("expires_in")

        try:
            if isinstance(expires_raw, (int, float)):
                seconds = float(expires_raw)
                if 0 < seconds < 60 * 60 * 24 * 7:
                    expires_at = datetime.utcnow() + timedelta(seconds=seconds)
            elif isinstance(expires_raw, str):
                safe_raw = expires_raw.replace("Z", "+00:00")
                expires_at = datetime.fromisoformat(safe_raw)
        except Exception:
            pass

        _gigachat_token_cache.update({"token": token, "expires_at": expires_at})
        return token, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Ошибка получения токена ГигаЧат: {exc}"


def _call_gigachat(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    try:
        from openai import DefaultHttpxClient, OpenAI  # type: ignore
    except Exception:
        return None, "Модуль openai не установлен. Установите его через pip."

    token, token_err = get_gigachat_token()
    if not token:
        return None, token_err or "Не удалось получить токен ГигаЧат"

    base_url = os.getenv("GIGACHAT_API_BASE", "https://gigachat.devices.sberbank.ru/api/v1")
    model = os.getenv("GIGACHAT_MODEL", "GigaChat")
    temperature = float(os.getenv("GIGACHAT_TEMPERATURE", "0.65"))
    verify = _gigachat_verify_tls()

    try:
        client = OpenAI(
            api_key=token,
            base_url=base_url,
            http_client=DefaultHttpxClient(verify=verify),
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=800,
        )
        content = response.choices[0].message.content
        return content, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Ошибка запроса ГигаЧат: {exc}"


def _build_prompt(payload: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    history_lines = []
    for idx, review in enumerate(history, start=1):
        history_lines.append(
            f"{idx}. {review.get('created_date', '—')} • "
            f"{review.get('course_name', 'Курс не указан')} • "
            f"оценка {review.get('rating', '?')}: {review.get('review_text', '').strip()}"
        )
    history_text = "\n".join(history_lines) if history_lines else "История не найдена."

    user_content = f"""
Ты — методист и педагог, пишущий официальные отзывы по ученикам.
Твоя задача — подготовить развернутый официальный отзыв на 8-12 предложений.

Данные формы:
- Студент: {payload.get('studentName', 'Неизвестно')} (id: {payload.get('studentId', '—')})
- Курс: {payload.get('courseName', 'Неизвестно')} (id: {payload.get('courseId', '—')})
- Преподаватель: {payload.get('teacherName', '—')}
- Текущая оценка (радио): {payload.get('rating', '—')}
- Черновик преподавателя: {payload.get('comment', '').strip() or '—'}

Последние отзывы на студента (от новых к старым):
{history_text}

Сформируй официальный отзыв:
1) Пиши в деловом стиле, избегай разговорных конструкций.
2) Покажи динамику прогресса ученика, опираясь на историю: что улучшилось, что требует внимания.
3) Добавь конкретику по задачам/темам, с которыми работает студент.
4) Закончь мягким рекомендационным выводом и пожеланием.
Верни только текст отзыва, без служебных пояснений.
"""
    return [
        {"role": "system", "content": "Ты пишешь официальные развёрнутые отзывы преподавателя."},
        {"role": "user", "content": user_content},
    ]


def _mock_review(payload: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    trend = "Данных о прогрессе недостаточно."
    if history:
        latest = history[0]
        trend = (
            f"Последний зафиксированный прогресс ({latest.get('created_date', '—')}): "
            f"{latest.get('review_text', '').strip()}"
        )
    return (
        f"Черновик от преподавателя: {payload.get('comment', '').strip() or '—'}. "
        f"Студент {payload.get('studentName', '—')} (курс {payload.get('courseName', '—')}) "
        f"имеет оценку {payload.get('rating', '—')}. "
        f"{trend} Новая версия отзыва: студент показывает устойчивый прогресс, "
        f"выполняет задания в срок и демонстрирует интерес к курсу. "
        f"Рекомендуется сохранить темп и уделять внимание самостоятельной практике."
    )


def generate_review(payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    history = get_last_reviews(payload.get("studentName", ""))
    prompt = _build_prompt(payload, history)
    draft, error = _call_gigachat(prompt)
    if not draft:
        draft = _mock_review(payload, history)
    return draft, history, error


def _save_generated_review(record: Dict[str, Any]) -> None:
    existing: List[Dict[str, Any]] = []
    if GENERATED_PATH.exists():
        existing = json.loads(GENERATED_PATH.read_text(encoding="utf-8") or "[]")
    existing.append(record)
    GENERATED_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_review_id(existing: List[Dict[str, Any]]) -> str:
    numbers = []
    for row in existing:
        raw_id = str(row.get("review_id", "")).strip()
        if raw_id.isdigit():
            numbers.append(int(raw_id))
    return str(max(numbers) + 1 if numbers else 1)


def _guess_sentiment(rating: int) -> str:
    if rating >= 5:
        return "positive"
    if rating <= 3:
        return "negative"
    return "neutral"


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def root() -> Any:
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/generate-review", methods=["POST"])
def api_generate_review() -> Any:
    data = request.get_json(silent=True) or {}
    required = ["studentName", "courseName", "comment", "rating"]
    missing = [field for field in required if not data.get(field)]
    if missing:
        return jsonify({"error": f"Не хватает данных: {', '.join(missing)}"}), 400

    draft, history, llm_error = generate_review(data)
    return jsonify(
        {
            "draftReview": draft,
            "history": history,
            "llmError": llm_error,
        }
    )


@app.route("/api/confirm-review", methods=["POST"])
def api_confirm_review() -> Any:
    data = request.get_json(silent=True) or {}
    review_text = data.get("reviewText")
    if not review_text:
        return jsonify({"error": "Нет текста отзыва"}), 400

    reviews = load_reviews()
    rating_raw = data.get("rating")
    try:
        rating_val = int(rating_raw)
    except Exception:
        rating_val = 0

    record = {
        "review_id": _next_review_id(reviews),
        "review_text": review_text,
        "student_name": data.get("studentName"),
        "teacher_name": data.get("teacherName"),
        "course_name": data.get("courseName"),
        "created_date": datetime.utcnow().date().isoformat(),
        "rating": rating_val,
        "sentiment": _guess_sentiment(rating_val),
        "word_count": len(review_text.split()),
    }

    reviews.append(record)
    _save_reviews(reviews)

    saved_record = {
        **record,
        "generated_at": datetime.utcnow().isoformat(),
        "llm_error": data.get("llmError"),
        "source": "llm",
        "draft_comment": data.get("comment"),
    }
    _save_generated_review(saved_record)

    history = get_last_reviews(record["student_name"])

    return jsonify({"status": "saved_to_csv", "record": saved_record, "history": history})


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
