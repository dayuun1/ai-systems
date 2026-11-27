import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

app = FastAPI(title="D&D 5e Encounter & NPC Builder")


class GenerateRequest(BaseModel):
    mode: str = Field(..., pattern="^(encounter|npc|rules)$")
    model: str = "qwen2.5:7b"
    temperature: float = Field(0.65, ge=0.0, le=1.5)
    max_tokens: int = Field(950, ge=100, le=4096)
    num_ctx: int = Field(4096, ge=1024, le=32768)

    party_level: Optional[int] = 2
    party_size: Optional[int] = 4
    difficulty: Optional[str] = "Середня (Medium)"
    biome: Optional[str] = "Закинутий портовий склад у тумані"
    party_notes: Optional[str] = ""

    npc_race: Optional[str] = "Людина"
    npc_role: Optional[str] = "Контрабандист-зброяр"
    npc_mood: Optional[str] = ""
    
    rule_question: Optional[str] = ""


def build_messages(req: GenerateRequest) -> list[dict]:
    if req.mode == "encounter":
        system_prompt = (
            "Ти — досвідчений Майстер Підземель (Dungeon Master) для настільної рольової гри D&D 5e. "
            "Відповідай виключно українською мовою. Підбирай монстрів відповідно до Challenge Rating (CR) "
            "та бюджету досвіду (XP) під рівень і кількість гравців, щоб бій був чесним і цікавим. "
            "Не вигадуй неіснуючих правил 5e."
        )
        user_prompt = (
            f"Згенеруй бойовий енкаунтер для D&D 5e:\n"
            f"- Рівень партії: {req.party_level}\n"
            f"- Кількість гравців: {req.party_size}\n"
            f"- Бажана складність: {req.difficulty}\n"
            f"- Локація: {req.biome}\n"
            f"- Додаткові умови: {req.party_notes.strip() or 'немає'}\n\n"
            f"Структура відповіді:\n"
            f"1. **Зав'язка та опис арени** (2–3 речення атмосфери + 2 інтерактивні об'єкти на полі бою).\n"
            f"2. **Склад ворогів і статблоки** (назва, кількість, CR, AC, HP, основна атака та бонус влучання).\n"
            f"3. **Тактика ворогів** (як вони діють у перші 2 раунди).\n"
            f"4. **Нагорода та трофеї** (золото, витратні предмети або зачіпка для сюжету)."
        )
    elif req.mode == "npc":
        system_prompt = (
            "Ти — сценарист та Майстер Підземель D&D 5e. Створюй колоритних, живих NPC українською мовою, "
            "яких зручно одразу відігравати за столом."
        )
        user_prompt = (
            f"Створи детального NPC для D&D 5e:\n"
            f"- Раса: {req.npc_race}\n"
            f"- Роль / професія: {req.npc_role.strip() or 'мандрівний торговець'}\n"
            f"- Характер і мотивація: {req.npc_mood.strip() or 'на твій вибір'}\n\n"
            f"Структура відповіді:\n"
            f"1. **Ім'я, зовнішність та манера мовлення** (1 яскрава деталь).\n"
            f"2. **Мотивація та таємниця** (чого хоче і що приховує від гравців).\n"
            f"3. **3 готові репліки для відіграшу** (привітання, торг/переговори, загроза).\n"
            f"4. **Короткий статблок** (AC, HP, ключові навички та спорядження)."
        )
    else:
        system_prompt = (
            "Ти — суворий арбітр правил D&D 5e (Player's Handbook та Sage Advice Compendium). "
            "Відповідай українською мовою чітко, посилаючись на офіційні механіки.\n"
            "Ключові правила Action Economy (PHB):\n"
            "1. У свій хід персонаж має Action, Bonus Action (якщо є вміння/закляття), Movement та 1 Reaction на раунд (можна використати і у свій хід, і в чужий — Bonus Action та Reaction НЕ взаємовиключні).\n"
            "2. Правило закляття бонусною дією (Bonus Action Spell, PHB розд. 10): якщо ви наклали закляття бонусною дією (наприклад, Misty Step), ви не можете накладати інші закляття основною дією (Action) у цей хід, окрім замовлянь (cantrips) з часом накладання 1 дія. На закляття реакцією (Reaction, наприклад Counterspell) це обмеження НЕ поширюється.\n"
            "Спочатку дай пряму відповідь (Так / Ні / Конкретне число), потім поясни покроково."
        )
        user_prompt = f"Розсудь ігрову ситуацію за правилами D&D 5e:\n\n{req.rule_question.strip()}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@app.get("/api/health")
def health():
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        return {"ok": True, "models": names}
    except Exception as exc:
        return {"ok": False, "models": ["qwen2.5:7b"], "error": str(exc)}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.mode == "rules" and not (req.rule_question or "").strip():
        def err_gen():
            yield "Помилка: введіть опис ігрової ситуації для перевірки правил."
        return StreamingResponse(err_gen(), media_type="text/plain; charset=utf-8")

    messages = build_messages(req)

    def token_stream():
        try:
            stream = client.chat.completions.create(
                model=req.model,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                extra_body={"options": {"num_ctx": req.num_ctx}},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield (
                f"\n\n[ПОМИЛКА OLLAMA]: {exc}\n"
                f"Перевірте, чи запущено сервер (ollama serve) та чи завантажена модель '{req.model}'."
            )

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).with_name("index.html")
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)