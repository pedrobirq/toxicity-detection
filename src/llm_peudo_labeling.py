import os
import re
import json
import random
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Set

from datasets import load_dataset
from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    APIError,
)
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm


# =====================================================================
# 1. Схемы валидации данных (Pydantic)
# =====================================================================

class LLMSpan(BaseModel):
    span: str = Field(description="Точная подстрока из токсичного текста")
    category: str = Field(
        description="Одна из категорий: profanity, insult, derogation_mockery, threat_violence, harassment_doxxing, hate_speech"
    )

class LLMResponse(BaseModel):
    spans: List[LLMSpan] = Field(default_factory=list)

class Entity(BaseModel):
    span: str
    start: int
    end: int
    category: str

class AnnotatedSample(BaseModel):
    id: int
    text: str
    neutral_reference: str
    entities: List[Entity]


# =====================================================================
# 2. Системный промпт и таксономия
# =====================================================================

SYSTEM_PROMPT = """Ты — эксперт по лингвистической экспертизе текстов в интернете. 
Твоя задача — найти в предложенном токсичном комментарии все спаны (фрагменты), содержащие токсичные языковые паттерны, и соотнести их с одной из категорий.

Тебе предоставляется токсичный комментарий и его нейтрализованный аналог (детоксикация). 
Используй нейтральный аналог как ориентир того, какие слова и паттерны были изменены или удалены.

Категории:
1. `profanity` — обсценная лексика (мат), междометия, ругательства общего плана без прямой адресации личности ("блядь", "какого хуя").
2. `insult` — прямые оскорбления личности, внешности, ума ("петух", "дебил", "биомусор").
3. `derogation_mockery` — язвительное коверкание слов, уничижительные ярлыки, насмешки ("херой", "мамкин эксперт", "рашка").
4. `threat_violence` — угрозы расправой, вредом, пожелания смерти ("тебе пиздец", "чтоб ты сдох").
5. `harassment_doxxing` — призывы к деанонимизации, травле, публикациям данных ("деанон", "слить номер").
6. `hate_speech` — язык вражды, ксенофобия, шовинизм по расовому, национальному, гендерному признаку.

Правила:
- Выделяй точную подстроку так, как она написана в токсичном тексте.
- Не включай знаки пунктуации, если они не являются частью смайла или оскорбления.
- Если токсичных спанов нет, верни пустой список spans: [].
- Ответ СТРОГО в формате JSON:
{"spans": [{"span": "...", "category": "..."}]}
"""


# =====================================================================
# 3. Вычисление точных смещений (Start/End)
# =====================================================================

def align_spans(text: str, llm_spans: List[LLMSpan]) -> List[Entity]:
    """Находит строгие индексы символов для подстрок, возвращенных LLM."""
    entities: List[Entity] = []
    cursor = 0
    
    for item in llm_spans:
        span_clean = item.span.strip()
        if not span_clean:
            continue
            
        pattern = re.escape(span_clean)
        # 1. Поиск вперед по тексту от курсора
        match = re.search(pattern, text[cursor:], flags=re.IGNORECASE)
        
        if match:
            start = cursor + match.start()
            end = cursor + match.end()
            entities.append(Entity(
                span=text[start:end],
                start=start,
                end=end,
                category=item.category
            ))
            cursor = end
        else:
            # 2. Резервный поиск с начала строки
            match_global = re.search(pattern, text, flags=re.IGNORECASE)
            if match_global:
                entities.append(Entity(
                    span=text[match_global.start():match_global.end()],
                    start=match_global.start(),
                    end=match_global.end(),
                    category=item.category
                ))
    return entities


# =====================================================================
# 4. Retry-логика с экспоненциальным бэкоффом и шумом (Jitter)
# =====================================================================

async def completion_with_backoff(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    max_retries: int = 6,
    base_delay: float = 3.0,
    max_delay: float = 60.0,
    **kwargs: Any
) -> Optional[Any]:
    """Выполняет запрос к Chat Completions с бэкоффом при ошибках 429 и 5xx."""
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            if attempt == max_retries - 1:
                print(f"\n[Критическая ошибка] Превышен лимит попыток ({max_retries}): {e}")
                raise e

            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.5, 2.0)
            print(f"\n[Retry {attempt + 1}/{max_retries}] Код: {type(e).__name__}. Пауза {delay:.1f} сек...")
            await asyncio.sleep(delay)

        except APIError as e:
            print(f"\n[Фатальная ошибка API {e.status_code}]: {e.message}")
            raise e

    return None


# =====================================================================
# 5. Сохранение прогресса и файловые операции
# =====================================================================

def load_processed_ids(file_path: Path) -> Set[int]:
    """Считывает id ранее обработанных строк для поддержки возобновления."""
    if not file_path.exists():
        return set()
    
    processed_ids = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "id" in record:
                    processed_ids.add(record["id"])
            except json.JSONDecodeError:
                continue
    return processed_ids


async def append_to_jsonl(file_path: Path, sample: AnnotatedSample, file_lock: asyncio.Lock):
    """Потокобезопасная дозапись с принудительным сбросом буфера."""
    line = json.dumps(sample.model_dump(), ensure_ascii=False) + "\n"
    async with file_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


# =====================================================================
# 6. Обработка одной записи
# =====================================================================

async def process_and_save_sample(
    client: AsyncOpenAI,
    sample_id: int,
    toxic_text: str,
    neutral_text: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
    output_path: Path,
    file_lock: asyncio.Lock,
) -> bool:
    user_prompt = f"Токсичный комментарий: {toxic_text}\nНейтральный аналог: {neutral_text}"
    
    async with semaphore:
        try:
            response = await completion_with_backoff(
                client=client,
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_retries=6,
                base_delay=3.0
            )
            
            if response is None:
                return False
                
            raw_content = response.choices[0].message.content
            parsed = LLMResponse.model_validate_json(raw_content)
            aligned_entities = align_spans(toxic_text, parsed.spans)
            
            sample = AnnotatedSample(
                id=sample_id,
                text=toxic_text,
                neutral_reference=neutral_text,
                entities=aligned_entities
            )
            
            await append_to_jsonl(output_path, sample, file_lock)
            return True
            
        except Exception as e:
            print(f"\n[Ошибка записи #{sample_id}]: {e}")
            return False


# =====================================================================
# 7. Точка входа пайплайна
# =====================================================================

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Переменная окружения OPENROUTER_API_KEY не задана.")
        
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Toxic Spans Labeling",
        }
    )
    
    model_name = "qwen/qwen-2.5-72b-instruct:free"
    output_path = Path("ru_paradetox_toxic_spans.jsonl")
    
    # 1. Проверяем сохраненный прогресс
    processed_ids = load_processed_ids(output_path)
    print(f"Найдено ранее размеченных записей: {len(processed_ids)}")
    
    # 2. Загружаем датасет
    print("Загрузка датасета s-nlp/ru_paradetox...")
    dataset = load_dataset("s-nlp/ru_paradetox", split="train")
    
    # 3. Фильтруем очередь задач
    pending_tasks_data = []
    for idx, row in enumerate(dataset):
        if idx in processed_ids:
            continue
        toxic_text = row["ru_toxic_comment"]
        neutral_text = row.get("ru_neutral_comment") or ""
        pending_tasks_data.append((idx, toxic_text, neutral_text))
        
    total_pending = len(pending_tasks_data)
    print(f"Осталось разметить: {total_pending} из {len(dataset)}")
    
    if total_pending == 0:
        print("Датасет уже полностью размечен.")
        return

    # 4. Контроль конкурентности
    # Для бесплатного эндпоинта рекомендуется держать 1-2 параллельных запроса
    concurrency_limit = 2
    semaphore = asyncio.Semaphore(concurrency_limit)
    file_lock = asyncio.Lock()
    
    tasks = [
        process_and_save_sample(
            client=client,
            sample_id=sample_id,
            toxic_text=toxic,
            neutral_text=neutral,
            semaphore=semaphore,
            model_name=model_name,
            output_path=output_path,
            file_lock=file_lock
        )
        for sample_id, toxic, neutral in pending_tasks_data
    ]
    
    print(f"Запуск разметки с параллелизмом = {concurrency_limit}...")
    results = await tqdm.gather(*tasks)
    
    success_count = sum(1 for res in results if res)
    print(f"\nСессия завершена. Успешно записано: {success_count}/{total_pending}")
    print(f"Результаты сохранены в: {output_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())