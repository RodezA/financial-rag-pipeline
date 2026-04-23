from unittest.mock import MagicMock, patch

from src.generation.llm_client import LLMClient


def _make_mock_response(answer_text="$412.7 million"):
    response = MagicMock()
    response.content = [MagicMock(text=answer_text)]
    response.usage.input_tokens = 500
    response.usage.output_tokens = 50
    response.usage.cache_creation_input_tokens = 480
    response.usage.cache_read_input_tokens = 0
    return response


@patch("src.generation.llm_client.anthropic.Anthropic")
def test_generate_returns_answer_and_usage(MockAnthropic):
    MockAnthropic.return_value.messages.create.return_value = _make_mock_response()

    client = LLMClient()
    result = client.generate("What was revenue?", [{"text": "Revenue $412.7M", "source": "doc.txt"}])

    assert "answer" in result
    assert "usage" in result
    assert result["answer"] == "$412.7 million"


@patch("src.generation.llm_client.anthropic.Anthropic")
def test_generate_system_prompt_has_cache_control(MockAnthropic):
    mock_create = MockAnthropic.return_value.messages.create
    mock_create.return_value = _make_mock_response()

    client = LLMClient()
    client.generate("question?", [{"text": "context", "source": "doc.txt"}])

    call_kwargs = mock_create.call_args.kwargs
    system = call_kwargs["system"]

    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}


@patch("src.generation.llm_client.anthropic.Anthropic")
def test_generate_context_block_has_cache_control_question_block_does_not(MockAnthropic):
    mock_create = MockAnthropic.return_value.messages.create
    mock_create.return_value = _make_mock_response()

    client = LLMClient()
    client.generate("What is revenue?", [{"text": "Revenue $412.7M", "source": "doc.txt"}])

    call_kwargs = mock_create.call_args.kwargs
    content = call_kwargs["messages"][0]["content"]

    assert len(content) == 2
    # First block = context: must have cache_control
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    # Second block = question: must NOT have cache_control
    assert "cache_control" not in content[1]


@patch("src.generation.llm_client.anthropic.Anthropic")
def test_generate_question_appears_in_final_block(MockAnthropic):
    mock_create = MockAnthropic.return_value.messages.create
    mock_create.return_value = _make_mock_response()

    client = LLMClient()
    client.generate("What is the net charge-off rate?", [{"text": "NCO: 1.42%", "source": "doc.txt"}])

    call_kwargs = mock_create.call_args.kwargs
    content = call_kwargs["messages"][0]["content"]
    question_block_text = content[1]["text"]

    assert "What is the net charge-off rate?" in question_block_text


@patch("src.generation.llm_client.anthropic.Anthropic")
def test_generate_usage_fields_present(MockAnthropic):
    MockAnthropic.return_value.messages.create.return_value = _make_mock_response()

    client = LLMClient()
    result = client.generate("q?", [{"text": "c", "source": "d.txt"}])

    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        assert key in result["usage"]
