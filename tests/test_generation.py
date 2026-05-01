from unittest.mock import MagicMock, patch

from src.generation.llm_client import LLMClient


def _make_mock_response(answer_text="$412.7 million"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = answer_text
    response.usage.prompt_tokens = 500
    response.usage.completion_tokens = 50
    return response


@patch("src.generation.llm_client.Groq")
def test_generate_returns_answer_and_usage(MockGroq):
    MockGroq.return_value.chat.completions.create.return_value = _make_mock_response()

    client = LLMClient()
    result = client.generate("What was revenue?", [{"text": "Revenue $412.7M", "source": "doc.txt"}])

    assert "answer" in result
    assert "usage" in result
    assert result["answer"] == "$412.7 million"


@patch("src.generation.llm_client.Groq")
def test_generate_system_prompt_sent(MockGroq):
    mock_create = MockGroq.return_value.chat.completions.create
    mock_create.return_value = _make_mock_response()

    client = LLMClient()
    client.generate("question?", [{"text": "context", "source": "doc.txt"}])

    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]

    # System prompt must be first message with role "system"
    assert messages[0]["role"] == "system"
    assert len(messages[0]["content"]) > 0


@patch("src.generation.llm_client.Groq")
def test_generate_context_in_user_message(MockGroq):
    mock_create = MockGroq.return_value.chat.completions.create
    mock_create.return_value = _make_mock_response()

    client = LLMClient()
    client.generate("What is revenue?", [{"text": "Revenue $412.7M", "source": "doc.txt"}])

    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]

    # User message must contain context and question
    user_content = messages[1]["content"]
    assert "Revenue $412.7M" in user_content
    assert "What is revenue?" in user_content


@patch("src.generation.llm_client.Groq")
def test_generate_question_appears_in_user_message(MockGroq):
    mock_create = MockGroq.return_value.chat.completions.create
    mock_create.return_value = _make_mock_response()

    client = LLMClient()
    client.generate("What is the net charge-off rate?", [{"text": "NCO: 1.42%", "source": "doc.txt"}])

    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]
    user_content = messages[1]["content"]

    assert "What is the net charge-off rate?" in user_content


@patch("src.generation.llm_client.Groq")
def test_generate_usage_fields_present(MockGroq):
    MockGroq.return_value.chat.completions.create.return_value = _make_mock_response()

    client = LLMClient()
    result = client.generate("q?", [{"text": "c", "source": "d.txt"}])

    for key in ("input_tokens", "output_tokens"):
        assert key in result["usage"]
