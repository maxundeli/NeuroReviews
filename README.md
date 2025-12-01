# Генератор отзывов для студентов

Короткое веб‑приложение на Flask: принимает данные по студенту, подтягивает историю отзывов из CSV и генерирует новый официальный отзыв через GigaChat (OpenAI‑совместимый интерфейс). Есть простая фронтенд-форма в `index.html`.

## Запуск
1) Python 3.10+  
2) Установить зависимости: `pip install -r requirements.txt`  
3) Создать `.env` в корне (пример ниже).  
4) Запустить сервер: `python server.py` (по умолчанию на `http://0.0.0.0:8000`).

## Переменные окружения
- `GIGACHAT_AUTH_KEY` — Basic auth ключ (выдается Сбером).  
- `GIGACHAT_SCOPE` — обычно `GIGACHAT_API_PERS`.  
- `GIGACHAT_API_BASE` — API базовый URL, пример: `https://gigachat.devices.sberbank.ru/api/v1`.  
- `GIGACHAT_AUTH_URL` — URL получения токена, пример: `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`.  
- `GIGACHAT_MODEL` — модель, например `GigaChat`.  
- `GIGACHAT_TEMPERATURE` — креативность (0–1).  
- `GIGACHAT_VERIFY_SSL` — `true/false`, выключает проверку сертификата (для самоподписанных).  
- `GIGACHAT_CA_BUNDLE` — путь к кастомному CA, если не хотите отключать SSL.

Сервер сам получает access token (живёт ~30 минут), кеширует его и обновляет.

## API
- `GET /` — отдает фронт.  
- `POST /api/generate-review` — вход: `studentName`, `courseName`, `comment`, `rating`, опционально `teacherName`, `studentId`, `courseId`. Ответ: сгенерированный черновик + история.  
- `POST /api/confirm-review` — сохраняет утвержденный текст в `reviews.csv` и `generated_reviews.json`.

## Данные
- `reviews.csv` — хранит отзывы (cp1251).  
- `generated_reviews.json` — логи сгенерированных отзывов.  
- История по студенту подтягивается из CSV и используется в промпте.
