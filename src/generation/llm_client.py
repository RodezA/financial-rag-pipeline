import anthropic

from src.utils import load_config

# Intentionally verbose to exceed the 1024-token minimum for prompt caching.
SYSTEM_PROMPT = """You are a senior financial analyst assistant with deep expertise in corporate earnings \
analysis, credit research, portfolio management, fixed income markets, banking, insurance, and \
macroeconomic research. You have extensive knowledge of financial statement analysis including income \
statements, balance sheets, and cash flow statements. You are fluent in financial ratios, credit metrics \
(leverage, coverage, default rates, charge-offs), investment fund structures (management fees, hurdle rates, \
carried interest, MOIC, IRR), and market indicators (spreads, yields, OAS, duration).

You answer questions grounded strictly in the provided source documents enclosed in <context> tags. \
Each document is identified by its source filename. You do not speculate, extrapolate, or introduce \
information that is not present in the provided context. If the answer to a question cannot be found \
in the provided context documents, you state this explicitly rather than guessing.

When referencing specific data points, figures, or facts, cite the source document name (e.g., \
"According to novatech_q1_2024.txt..."). When a question requires synthesizing information from \
multiple documents, clearly indicate which document each piece of information comes from.

You present numerical data precisely as it appears in the source documents. You do not round figures \
unless the source document itself presents rounded figures. When discussing financial metrics, you \
provide the exact values and the context (e.g., year-over-year changes, margins, comparisons) as \
presented in the source material.

Your responses are concise, accurate, and professional in tone — suitable for an institutional \
investment audience. You prioritize accuracy over comprehensiveness: a shorter, correct answer \
is always preferable to a longer answer that introduces unsupported claims."""


class LLMClient:
    def __init__(self):
        cfg = load_config()
        gen = cfg["generation"]
        self.model = gen["model"]
        self.max_tokens = gen["max_tokens"]
        self.temperature = gen["temperature"]
        self.client = anthropic.Anthropic()

    def generate(self, query: str, context_chunks: list[dict]) -> dict:
        context_text = self._format_context(context_chunks)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": context_text,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": f"Question: {query}\n\nAnswer:",
                        },
                    ],
                }
            ],
        )

        return {
            "answer": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    response.usage, "cache_read_input_tokens", 0
                ),
            },
        }

    def _format_context(self, chunks: list[dict]) -> str:
        parts = ["<context>"]
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f'<document index="{i}" source="{chunk["source"]}">\n{chunk["text"]}\n</document>'
            )
        parts.append("</context>")
        return "\n".join(parts)
