import json
import os
import re
import time
from typing import List

from django.core.management import BaseCommand, CommandError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from copilot.llm.provider import get_chat


class FengshuiAnswer(BaseModel):
    # Dấu ... (gọi là ellipsis) ở đây là một giá trị đặc biệt từ module builtins của Python, đại diện cho việc
    # field này bắt buộc phải có giá trị (required) và không có giá trị mặc định (no default value).
    # Nếu muốn có default, bạn thay ... bằng giá trị cụ thể, ví dụ Field("Mặc định").
    answer: str = Field(..., description="Câu trả lời ngắn gọn bằng tiếng Viêt, ~80 từ")
    citations: List[str] = Field(
        default_factory=list,
        description="Các khái niệm/nguồn liên quan (ví dụ: ngũ hành, bát quái,...)"
    )
    confidence: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Độ tin cậy từ 0.0 đến 1.0"
    )

class Command(BaseCommand):
    help = "Hỏi đáp phong thủy (LangChain + Ollama) có ép JSON"

    def add_arguments(self, parser):
        parser.add_argument("--q", dest="question", required=False,
                            default="Mệnh Kim hợp màu gì?", help="Câu hỏi tiếng Việt")

    def handle(self, *args, **opts):
        model = os.getenv("LLM_MODEL")
        llm = get_chat(model, temperature=0.0)
        self.stdout.write(f"LLM={llm.__class__.__name__} | model={model}")

        # Tạo parser + format instructions
        parser = PydanticOutputParser(pydantic_object=FengshuiAnswer)
        format_instructions = parser.get_format_instructions()

        # Chuẩn bị prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Bạn là chuyên gia phong thủy, hãy trả lời ngắn gọn, không bịa"
             " CHỈ TRẢ VỀ DỮ LIỆU THEO SCHEMA JSON được cung cấp"),
            ("human",
             "Câu hỏi: {question}\n\n"
             "Hãy tuân thủ nghiêm ngặt hướng dẫn định dạng sau:\n {format_instructions}")
        ])

        # Tạo chain:
        chain = prompt | llm | parser

        t0 = time.time()
        try:
            # Đây chính là tác dụng của parser
            res: FengshuiAnswer = chain.invoke({
                "question": opts["question"],
                "format_instructions": format_instructions
            })
        except Exception as e:
            self.stderr.write(self.style.WARNING(f"Parse lỗi, thử fallback: {e}"))
            raw = (prompt | llm).invoke({
                "question": opts["question"],
                "format_instructions": format_instructions
            }).content

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise CommandError("Không tìm thấy JSON trong câu trả lời")
            res = parser.parse(match.group(0))

        dt = time.time() - t0

        self.stdout.write(json.dumps(res.dict(), ensure_ascii=False, indent=2))
        self.stdout.write(self.style.SUCCESS(f"[{model}] took {dt:.2f}s"))
