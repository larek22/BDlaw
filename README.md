# BDlaw Desktop Toolkit

BDlaw — кроссплатформенное настольное приложение для импорта юридических документов, построения векторной базы норм и получения ответов GPT со строгими цитатами. Системные цели, UX и roadmap описаны в [docs/system_design.md](docs/system_design.md).

## Основные возможности

- Импорт PDF/DOCX/RTF/ODT/TXT/HTML/сканов с OCR, нормализация текста и выделение структуры закона (статья → часть → пункт → подпункт).
- Построение детерминированных идентификаторов норм, хранение метаданных об редакциях, поправках, решениях КС РФ и перекрестных ссылках.
- Индексация чанков в Qdrant с использованием эмбеддингов OpenAI `text-embedding-3-small` и плагинной архитектуры для альтернативных хранилищ.
- Семантический поиск и формирование ответов GPT-4.1/4.1-mini со строгими цитатами и источниками.
- Встроенный тест-раннер golden-set с метриками Recall@k, Precision@k, MRR@k, nDCG@k и Answer-PASS@k.
- Настольный GUI на PySide6 с экранами Home/Import/Preview/Index/Search/Answer/Tests/Settings.
- CLI-утилита для скриптовых сценариев (`bdlaw ingest/search/answer/tests`).

## Структура проекта

```
app/
  ui/                # PySide6 GUI
ingest/              # Импорт и нормализация исходных документов
parser/              # Разметка структуры норм и чанкинг
index/               # Эмбеддинги, векторное хранилище и индексация
search/              # Семантический поиск и фильтры
gpt/                 # Ответы GPT и валидатор
testsuite/           # Метрики качества Retrieval/Answer
settings/            # Конфигурация и работа с keyring
export/              # Экспорт в JSONL и снапшоты
utils/               # Общие утилиты
```

CLI и GUI используют единый `ApplicationPipelines`, обеспечивающий ingest → parse → chunk → embed → index.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Установите [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) и добавьте бинарь в PATH либо укажите путь в `config.yaml`. Для работы с RTF требуется установленный [Pandoc](https://pandoc.org/).

### Локальное хранилище Qdrant

```bash
docker compose up -d qdrant
```

Коллекция `laws` создаётся автоматически при первом запуске.

## Конфигурация и секреты

- Основные параметры см. в `config.yaml` (OCR, нормализация текста, чанкинг, эмбеддинги, Qdrant, модели LLM).
- API-ключи OpenAI хранятся через системный keyring. Пример использования:

```python
from bdlaw.settings.secrets import SecretStore
SecretStore().set("openai_api_key", "sk-...")
```

## CLI-команды

```bash
bdlaw ingest ./laws/gk.pdf \
  --law-code "ГК РФ" \
  --law-full-title "Гражданский кодекс РФ (часть первая)" \
  --version-id 2024-09-01 \
  --valid-from 2024-09-01 \
  --source-url https://publication.pravo.gov.ru/Document/View/...

bdlaw search --query "Проценты за пользование чужими денежными средствами" --law-code "ГК РФ" --k 8

bdlaw answer --query "Как взыскиваются проценты по ст. 395 ГК РФ?" --law-code "ГК РФ" --k 6

bdlaw tests golden_set.example.json --k 8
```

### Быстрый старт на демо-наборе

В репозитории лежит небольшой образец `data/samples/gk_excerpt.txt`, покрывающий статьи 12, 309, 310 и 395 ГК РФ, а также JSONL с готовыми нормами `data/samples/laws.sample.jsonl` (для изучения структуры payload).

```bash
# Импортируем и индексируем демо-текст
bdlaw ingest data/samples/gk_excerpt.txt \
  --law-code "ГК РФ" \
  --law-full-title "Гражданский кодекс РФ (часть первая)" \
  --version-id 2024-09-01 \
  --valid-from 2024-09-01 \
  --source-url https://publication.pravo.gov.ru/Document/View/0001202409010001

# Проверяем поиск и ответы
bdlaw search --query "проценты по денежному обязательству" --law-code "ГК РФ" --date 2024-10-01 --k 5
bdlaw answer --query "Какие способы защиты гражданских прав?" --law-code "ГК РФ" --date 2024-10-01 --k 8

# Запускаем golden-set (использует ст. 395 ГК РФ)
bdlaw tests golden_set.example.json --k 5
```

Если Qdrant уже содержит собственные данные, воспользуйтесь экспортом `JSONL` (`bdlaw.export.jsonl`) и снапшотами Qdrant для бэкапов.

## GUI

```bash
# Автоматически попытается поднять Qdrant через docker compose и запустить GUI
python start_bdlaw.py

# Пропустить автозапуск Qdrant (если сервис уже поднят либо Docker недоступен)
python start_bdlaw.py --no-compose

# Классический запуск через модуль (без автоподъёма Qdrant)
python -m bdlaw.app.ui
```

На Windows можно воспользоваться батником `start_bdlaw_gui.bat`: он ищет виртуальное окружение `.venv` и вызывает
`python start_bdlaw.py --no-compose`. Достаточно дважды щёлкнуть файл после установки зависимостей.

Приложение поднимает полноценный рабочий стол в тёмной теме: на домашнем экране крупные кнопки направляют по основным сценариям, а страницы Import/Index/Search/Tests снабжены подсказками и наглядными индикаторами прогресса. Выберите файлы, заполните всплывающий диалог с метаданными (юрисдикция, код закона, редакция) — индексирование запустится в фоне. Поиск, формирование ответа GPT и просмотр логов доступны сразу после завершения индексации. Горячие клавиши: `Ctrl+I` (Импорт), `Ctrl+Enter` (Индексировать), `Ctrl+F` (Поиск), `F5` (Тесты).

## Тесты

```bash
pytest
```

## Сборка .exe (PyInstaller)

```bash
pyinstaller --name bdlaw --noconsole --onefile -p . bdlaw/cli.py
```

Готовый бинарь будет в `dist/bdlaw.exe`. Для включения GUI добавьте входную точку `bdlaw_gui = bdlaw.app.ui.main:run_ui` в `pyproject.toml` и соберите отдельный `.spec`.

## Дополнительные материалы

- Шаблон golden-set: `golden_set.example.json`.
- Документ системы: [docs/system_design.md](docs/system_design.md).
