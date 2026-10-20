import os
import json
import re
from typing import List, Optional
from collections import defaultdict
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. Pydantic-схемы для валидации
# ---------------------------------------------------------
class ToxicSpanItem(BaseModel):
    span: str = Field(description="Точная подстрока из исходного текста, содержащая токсичный паттерн")
    category: str = Field(
        description="Категория: profanity, insult, derogation_mockery, threat_violence, harassment_doxxing, hate_speech"
    )

class LLMAnnotationResponse(BaseModel):
    toxic_patterns: List[ToxicSpanItem] = Field(default_factory=list)


# ---------------------------------------------------------
# 2. Системный промпт с таксономией и Few-Shot примерами
# ---------------------------------------------------------
SYSTEM_PROMPT = """Ты — эксперт по лингвистическому анализу токсичного контента в русском языке.
Твоя задача: найти в токсичном комментарии все спаны (фрагменты) токсичных паттернов и классифицировать их.
Тебе также даются нейтральные варианты перефразирования того же комментария — используй их как подсказку, чтобы понять, какие именно слова несут токсичность.

Таксономия классов:
1. profanity: обсценная лексика (мат), ругательства и междометия без прямой адресации ("блядь", "нахуй").
2. insult: прямое оскорбление личности или статуса ("петух", "дебил", "урод").
3. derogation_mockery: уничижение, язвительный сарказм, искажения слов ("херой", "мамкин эксперт", "рашка").
4. threat_violence: прямые и косвенные угрозы, призывы к насилию, пожелания смерти ("тебе пиздец", "чтоб ты сдох").
5. harassment_doxxing: травля, призывы к деанонимизации и сливу личных данных ("деанон этого...", "пробейте номер").
6. hate_speech: язык вражды, дискриминация по нации, расе, полу, религии, соц. статусу.

Правила:
- Выделяй только минимально необходимые слова или словосочетания (не захватывай лишние нейтральные слова).
- span должен совпадать с исходным текстом символ-в-символ.
- Если токсичных элементов нет, верни пустой список toxic_patterns.
- Ответ ОБЯЗАТЕЛЬНО должен быть валидным JSON строго по указанной схеме.
"""

# ---------------------------------------------------------
# 3. Вспомогательная функция для расчета смещений (offsets)
# ---------------------------------------------------------
def calculate_offsets(original_text: str, spans: List[ToxicSpanItem]) -> List[dict]:
    """
    Находит точные индексы start и end для каждого спана в исходном тексте,
    учитывая возможное повторение одинаковых слов.
    """
    enriched_spans = []
    current_search_pos = 0

    for item in spans:
        span_str = item.span.strip()
        if not span_str:
            continue
        
        # Поиск подстроки начиная с текущей позиции
        idx = original_text.find(span_str, current_search_pos)
        
        # Если не найдено вперед, пробуем найти с начала текста
        if idx == -1:
            idx = original_text.find(span_str)

        if idx != -1:
            start_pos = idx
            end_pos = idx + len(span_str)
            enriched_spans.append({
                "span": span_str,
                "start": start_pos,
                "end": end_pos,
                "category": item.category
            })
            current_search_pos = end_pos
        else:
            # Спан был галлюцинирован моделью или видоизменен
            print(f"[WARN] Спан '{span_str}' не найден в строке: '{original_text}'")

    return enriched_spans


# ---------------------------------------------------------
# 4. Основной пайплайн
# ---------------------------------------------------------
def run_pipeline(
    output_filepath: str = "ru_paradetox_annotated.jsonl",
    limit: Optional[int] = 50
):
    # Инициализация клиента OpenRouter (OpenAI-совместимый)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Задайте переменную окружения OPENROUTER_API_KEY")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("Загрузка датасета s-nlp/ru_paradetox...")
    dataset = load_dataset("s-nlp/ru_paradetox", split="train")

    # Группируем нейтральные варианты по уникальному токсичному тексту
    print("Дедупликация и подготовка данных...")
    toxic_to_neutrals = defaultdict(list)
    for row in dataset:
        toxic_text = row["toxic"]
        neutral_text = row["neutral"]
        if neutral_text not in toxic_to_neutrals[toxic_text]:
            toxic_to_neutrals[toxic_text].append(neutral_text)

    unique_toxic_items = list(toxic_to_neutrals.items())
    if limit:
        unique_toxic_items = unique_toxic_items[:limit]

    print(f"Всего уникальных текстов для разметки: {len(unique_toxic_items)}")

    processed_count = 0
    with open(output_filepath, "w", encoding="utf-8") as out_f:
        for idx, (toxic_text, neutrals) in enumerate(unique_toxic_items):
            user_content = (
                f"Токсичный комментарий: \"{toxic_text}\"\n"
                f"Нейтральные аналоги: {json.dumps(neutrals, ensure_ascii=False)}\n\n"
                f"Извлеки токсичные спаны в JSON-формате: "
                f'{{"toxic_patterns": [{{"span": "...", "category": "..."}}]}}'
            )

            try:
                # Запрос к Qwen 2.5 72B Instruct
                response = client.chat.completions.create(
                    model="qwen/qwen-2.5-72b-instruct:free",
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                )

                raw_json = response.choices[0].message.content
                parsed = LLMAnnotationResponse.model_validate_json(raw_json)
                
                # Расчет физических символьных индексов в строке
                spans_with_offsets = calculate_offsets(toxic_text, parsed.toxic_patterns)

                record = {
                    "text_id": f"paradetox_{idx}",
                    "text": toxic_text,
                    "neutral_references": neutrals,
                    "spans": spans_with_offsets
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                processed_count += 1
                print(f"[{processed_count}/{len(unique_toxic_items)}] Успешно: {toxic_text[:40]}...")

            except Exception as e:
                print(f"[ERROR] Ошибка при обработке записи {idx}: {e}")

    print(f"Разметка завершена! Результат сохранен в {output_filepath}")


if __name__ == "__main__":
    run_pipeline(output_filepath="ru_paradetox_annotated.jsonl", limit=20)