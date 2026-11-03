import re
import unicodedata


def inspection_text(text: str) -> str:
    """Нормализация только для проверки; исходные значения не меняются."""
    return "".join(char for char in unicodedata.normalize("NFKC", text)
                   if unicodedata.category(char) != "Cf")


MODEL_COMMAND = re.compile(
    r"\b(?:игнориру\w*|забудь\w*|отмени\w*).{0,80}\b(?:правил\w*|инструкц\w*|промпт\w*)"
    r"|\b(?:ignore|disregard|override).{0,80}\b(?:instructions?|rules?|prompts?|previous)"
    r"|\b(?:вставь\w*|включи\w*|добавь\w*|напиши\w*|выведи\w*|повтори\w*).{0,100}"
    r"\b(?:ответ\w*|готов\w*\s+описан\w*|фраз\w*|промпт\w*)"
    r"|\b(?:include|insert|print|repeat|append|write).{0,100}\b(?:response|answer|phrase|prompt)"
    r"|\bслужебн\w+\s+команд\w*"
    r"|\bинструкц\w+\s+для\s+(?:языков\w+\s+модел\w*|llm|ии|нейросет\w*)"
    r"|(?:^|\n)\s*(?:system|developer)\s*:"
    r"|<\|(?:im_start|start_header_id)\|>", re.IGNORECASE,
)


def has_model_command(text: str) -> bool:
    text = inspection_text(text)
    # Контекст управления ответом не является характеристикой прибора;
    # проверка не зависит от конкретного глагола («допиши», «перенеси» и т.п.).
    output_target = re.search(
        r"\b(?:в\s+(?:(?:итогов\w*|готов\w*|сво\w*|тво\w*)\s+)?(?:ответ\w*|описан\w*|текст\w*|карточк\w*)"
        r"|seo[-\s]?(?:описан\w*|заголов\w*))\b", text, re.IGNORECASE)
    output_content = re.search(r"\b(?:фраз\w*|маркер\w*|слов[оа]\w*|контрольн\w*)\b", text, re.IGNORECASE)
    return MODEL_COMMAND.search(text) is not None or bool(output_target and output_content)


def is_russian(text: str) -> bool:
    russian = len(re.findall(r"[а-яё]", text, re.IGNORECASE))
    latin = len(re.findall(r"[a-z]", text, re.IGNORECASE))
    return russian >= 3 and russian >= latin
