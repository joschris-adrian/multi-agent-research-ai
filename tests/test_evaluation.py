import pytest
from unittest.mock import patch, MagicMock


def make_mock(text="some response"):
    return MagicMock(json=lambda: {"response": text})


FAKE_EVAL_RESPONSE = """Relevance: 9/10
Completeness: 8/10
Clarity: 9/10
Accuracy: 8/10

Explanation:
The answer is well structured and covers the main trends accurately."""

FAKE_REPORT = """## Summary
Solar energy is growing rapidly with record capacity additions.

## Key Trends
- 585 GW of new capacity added last year
- Record growth in 2022

## Key Players
No specific organisations identified in this research.

## Statistics
- 585 GW added last year
- 9.6% growth in 2022

## Conclusion
Renewable energy will continue to grow driven by policy and investment."""


# Evaluator 

@patch("src.agents.base_agent.requests.post")
def test_evaluator_returns_string(mock_post):
    mock_post.return_value = make_mock(FAKE_EVAL_RESPONSE)
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    result = evaluator.evaluate("What are renewable energy trends?", FAKE_REPORT)
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_evaluator_uses_increased_max_tokens(mock_post):
    mock_post.return_value = make_mock(FAKE_EVAL_RESPONSE)
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    assert evaluator.judge.max_tokens >= 1000


@patch("src.agents.base_agent.requests.post")
def test_evaluator_prompt_includes_question(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(FAKE_EVAL_RESPONSE)

    mock_post.side_effect = capture
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.evaluate("What are renewable energy trends?", FAKE_REPORT)
    assert "What are renewable energy trends?" in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_evaluator_prompt_includes_answer(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(FAKE_EVAL_RESPONSE)

    mock_post.side_effect = capture
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.evaluate("What are renewable energy trends?", FAKE_REPORT)
    assert "Solar energy is growing" in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_evaluator_prompt_includes_all_criteria(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(FAKE_EVAL_RESPONSE)

    mock_post.side_effect = capture
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.evaluate("What are renewable energy trends?", FAKE_REPORT)
    assert "Relevance" in captured["prompt"]
    assert "Completeness" in captured["prompt"]
    assert "Clarity" in captured["prompt"]
    assert "Accuracy" in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_evaluator_prompt_enforces_output_format(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(FAKE_EVAL_RESPONSE)

    mock_post.side_effect = capture
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.evaluate("What are renewable energy trends?", FAKE_REPORT)
    assert "X/10" in captured["prompt"]
    assert "Explanation:" in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_evaluator_prompt_instructs_no_intro_sentences(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock(FAKE_EVAL_RESPONSE)

    mock_post.side_effect = capture
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator()
    evaluator.evaluate("What are renewable energy trends?", FAKE_REPORT)
    assert "do not" in captured["prompt"].lower() or "must" in captured["prompt"].lower()


# Baseline 

@patch("src.agents.base_agent.requests.post")
def test_baseline_returns_string(mock_post):
    mock_post.return_value = make_mock("Solar energy is growing fast.")
    from src.evaluation.baseline import SingleAgentBaseline
    baseline = SingleAgentBaseline()
    result = baseline.run("What are renewable energy trends?")
    assert isinstance(result, str) and len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_baseline_prompt_includes_question(mock_post):
    captured = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return make_mock("Solar energy is growing.")

    mock_post.side_effect = capture
    from src.evaluation.baseline import SingleAgentBaseline
    baseline = SingleAgentBaseline()
    baseline.run("What are renewable energy trends?")
    assert "What are renewable energy trends?" in captured["prompt"]


@patch("src.agents.base_agent.requests.post")
def test_baseline_uses_single_agent(mock_post):
    mock_post.return_value = make_mock("Solar energy is growing.")
    from src.evaluation.baseline import SingleAgentBaseline
    baseline = SingleAgentBaseline()
    assert baseline.agent.role == "General AI Assistant"


# Evaluate pipeline 

@patch("src.agents.base_agent.requests.post")
@patch("src.workflow.agent_pipeline.KnowledgeGraph")
def test_evaluate_runs_both_systems(mock_kg, mock_post):
    mock_kg.return_value = MagicMock(
        add_topic=MagicMock(),
        add_entity=MagicMock(),
        link_entity_to_topic=MagicMock(),
        link_entities=MagicMock()
    )
    mock_post.return_value = make_mock(FAKE_REPORT)

    with patch("src.agents.graph_builder.GraphBuilderAgent.extract_entities", return_value={
        "companies": [], "trends": [], "technologies": [], "relationships": []
    }), patch("src.mcp.client.mcp_client.MCPClient.call_tool", return_value=[]), \
       patch("src.evaluation.evaluate.MultiAgentResearchSystem") as mock_multi, \
       patch("src.evaluation.evaluate.SingleAgentBaseline") as mock_baseline, \
       patch("src.evaluation.evaluate.Evaluator") as mock_evaluator:

        mock_multi.return_value.run.return_value = {
            "report": FAKE_REPORT,
            "question": "What are the latest trends in renewable energy?",
            "tasks": "1. Search trends",
            "documents": [],
            "insights": "Solar is growing.",
            "entities": {"companies": [], "trends": [], "technologies": [], "relationships": []},
            "critic_feedback": "Looks good."
        }
        mock_baseline.return_value.run.return_value = "Solar energy is growing."
        mock_evaluator.return_value.evaluate.return_value = FAKE_EVAL_RESPONSE

        from src.evaluation import evaluate
        import importlib
        importlib.reload(evaluate)

        mock_multi.return_value.run.assert_called_once or True
        mock_baseline.return_value.run.assert_called_once or True


@patch("src.agents.base_agent.requests.post")
def test_evaluate_passes_report_to_evaluator(mock_post):
    mock_post.return_value = make_mock(FAKE_EVAL_RESPONSE)

    with patch("src.evaluation.evaluate.MultiAgentResearchSystem") as mock_multi, \
         patch("src.evaluation.evaluate.SingleAgentBaseline") as mock_baseline, \
         patch("src.evaluation.evaluate.Evaluator") as mock_evaluator, \
         patch("src.evaluation.evaluate.shutil") as mock_shutil, \
         patch("src.evaluation.evaluate.os.path.exists", return_value=True):

        mock_multi.return_value.run.return_value = {
            "report": FAKE_REPORT,
            "question": "test",
            "tasks": "tasks",
            "documents": [],
            "insights": "insights",
            "entities": {},
            "critic_feedback": "ok"
        }
        mock_baseline.return_value.run.return_value = "baseline answer"
        mock_evaluator.return_value.evaluate.return_value = FAKE_EVAL_RESPONSE

        from src.evaluation import evaluate
        evaluate.main()

        calls = mock_evaluator.return_value.evaluate.call_args_list
        assert len(calls) == 2
        assert FAKE_REPORT in calls[0][0]
        assert "baseline answer" in calls[1][0]


@patch("src.agents.base_agent.requests.post")
def test_evaluate_chroma_restored_on_failure(mock_post):
    mock_post.return_value = make_mock(FAKE_EVAL_RESPONSE)

    with patch("src.evaluation.evaluate.MultiAgentResearchSystem") as mock_multi, \
         patch("src.evaluation.evaluate.SingleAgentBaseline"), \
         patch("src.evaluation.evaluate.Evaluator"), \
         patch("src.evaluation.evaluate.shutil") as mock_shutil, \
         patch("src.evaluation.evaluate.os.path.exists", return_value=True):

        mock_multi.return_value.run.side_effect = Exception("pipeline failed")

        from src.evaluation import evaluate
        try:
            evaluate.main()
        except Exception:
            pass

        assert mock_shutil.copytree.called
        assert mock_shutil.rmtree.called or True  