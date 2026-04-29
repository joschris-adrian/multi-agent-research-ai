import os
import json
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.finetuning


# ── dataset ───────────────────────────────────────────────────────────────────

def test_dataset_exists():
    assert os.path.exists("training/dataset.json"), \
        "training/dataset.json not found — run generate_training_data.py first"


def test_dataset_has_required_keys():
    with open("training/dataset.json") as f:
        data = json.load(f)
    assert len(data) > 0
    for example in data:
        assert "instruction" in example
        assert "output" in example
        assert len(example["instruction"]) > 0
        assert len(example["output"]) > 0


def test_dataset_has_enough_examples():
    with open("training/dataset.json") as f:
        data = json.load(f)
    assert len(data) >= 3


# ── adapter ───────────────────────────────────────────────────────────────────

def test_adapter_exists():
    assert os.path.exists("models/lora-adapter"), \
        "models/lora-adapter not found — run training/finetune.py first"


def test_adapter_has_required_files():
    adapter_path = "models/lora-adapter"
    if not os.path.exists(adapter_path):
        pytest.skip("adapter not trained yet")
    files = os.listdir(adapter_path)
    assert any("adapter" in f for f in files), \
        f"no adapter files found in {adapter_path}, got: {files}"


# ── peft model (mocked at sys.modules level to avoid torch DLL issues) ────────

def make_torch_mock():
    torch_mock = MagicMock()
    torch_mock.float32 = "float32"
    torch_mock.no_grad.return_value.__enter__ = lambda s: s
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    tensor_mock = MagicMock()
    tensor_mock.shape = [1, 3]
    torch_mock.tensor.return_value = tensor_mock
    return torch_mock


def test_finetuned_writer_loads():
    torch_mock = make_torch_mock()
    transformers_mock = MagicMock()
    peft_mock = MagicMock()

    with patch.dict("sys.modules", {
        "torch": torch_mock,
        "transformers": transformers_mock,
        "peft": peft_mock,
    }), patch("os.path.exists", return_value=True):
        import importlib
        import src.models.peft_model as pm
        importlib.reload(pm)
        writer = pm.FineTunedWriter()
        assert writer is not None


def test_finetuned_writer_generate():
    torch_mock = make_torch_mock()
    transformers_mock = MagicMock()
    peft_mock = MagicMock()

    mock_tok = MagicMock()
    mock_tok.eos_token_id = 2
    mock_tok.decode.return_value = "Generated report"
    mock_tok.return_value = {"input_ids": MagicMock(shape=[1, 3])}
    transformers_mock.AutoTokenizer.from_pretrained.return_value = mock_tok

    mock_mdl = MagicMock()
    mock_mdl.generate.return_value = MagicMock()
    peft_mock.PeftModel.from_pretrained.return_value = mock_mdl

    with patch.dict("sys.modules", {
        "torch": torch_mock,
        "transformers": transformers_mock,
        "peft": peft_mock,
    }), patch("os.path.exists", return_value=True):
        import importlib
        import src.models.peft_model as pm
        importlib.reload(pm)
        writer = pm.FineTunedWriter()
        result = writer.generate("Write a report on solar energy")
        assert isinstance(result, str)


def test_finetuned_writer_raises_if_no_adapter():
    torch_mock = make_torch_mock()
    transformers_mock = MagicMock()
    peft_mock = MagicMock()

    with patch.dict("sys.modules", {
        "torch": torch_mock,
        "transformers": transformers_mock,
        "peft": peft_mock,
    }), patch("os.path.exists", return_value=False):
        import importlib
        import src.models.peft_model as pm
        importlib.reload(pm)
        with pytest.raises(FileNotFoundError):
            pm.FineTunedWriter()


# ── writer agent ──────────────────────────────────────────────────────────────

@patch("src.agents.base_agent.requests.post")
def test_writer_falls_back_to_ollama_by_default(mock_post):
    mock_post.return_value = MagicMock(
        json=lambda: {"response": "Report from Ollama"}
    )
    with patch.dict(os.environ, {"USE_FINETUNED": "0"}):
        import importlib
        import src.agents.writer as writer_module
        importlib.reload(writer_module)
        writer = writer_module.WriterAgent()
        result = writer.write_report("some insights")
        assert isinstance(result, str)
        assert len(result) > 0


@patch("src.agents.base_agent.requests.post")
def test_writer_uses_finetuned_when_env_set(mock_post):
    mock_post.return_value = MagicMock(json=lambda: {"response": "fallback"})
    mock_finetuned = MagicMock()
    mock_finetuned.generate.return_value = "Fine-tuned report"

    torch_mock = make_torch_mock()
    transformers_mock = MagicMock()
    peft_mock = MagicMock()

    with patch.dict("sys.modules", {
        "torch": torch_mock,
        "transformers": transformers_mock,
        "peft": peft_mock,
    }), patch.dict(os.environ, {"USE_FINETUNED": "1"}), \
        patch("os.path.exists", return_value=True):

        import importlib
        import src.models.peft_model as pm
        importlib.reload(pm)
        pm.FineTunedWriter = MagicMock(return_value=mock_finetuned)

        import src.agents.writer as writer_module
        importlib.reload(writer_module)
        w = writer_module.WriterAgent()
        result = w.write_report("some insights")
        assert result == "Fine-tuned report"
