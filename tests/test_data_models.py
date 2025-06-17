import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from truss.data_models import Message, ToolCall, ToolCallResult, LLMConfig, AgentConfig, AgentMemory, AgentWorkflowInput, AgentWorkflowOutput
from uuid import uuid4


def test_message_serialization_roundtrip():
    msg = Message(role="user", content="Hello world")
    json_str = msg.model_dump_json()
    parsed = Message.model_validate_json(json_str)
    assert parsed == msg


def test_tool_call_roundtrip():
    tool_call = ToolCall(name="add_numbers", arguments={"a": 1, "b": 2})
    json_str = tool_call.model_dump_json()
    parsed = ToolCall.model_validate_json(json_str)
    assert parsed == tool_call


def test_tool_call_result_roundtrip():
    tool_call = ToolCall(name="fake", arguments={})
    result = ToolCallResult(tool_call_id=tool_call.id, content={"ok": True})
    json_str = result.model_dump_json()
    parsed = ToolCallResult.model_validate_json(json_str)
    assert parsed == result


def test_invalid_role_raises_validation_error():
    with pytest.raises(ValueError):
        Message(role="invalid", content="oops")


def test_llm_config_roundtrip():
    cfg = LLMConfig(model_name="gpt-4o", temperature=0.5, max_tokens=256)
    json_str = cfg.model_dump_json()
    parsed = LLMConfig.model_validate_json(json_str)
    assert parsed == cfg


def test_agent_config_roundtrip():
    cfg = AgentConfig(
        name="Test Agent",
        system_prompt="You are a helpful assistant",
        llm_config=LLMConfig(model_name="gpt-3.5-turbo"),
        tools=["calculator", "search"],
    )
    json_str = cfg.model_dump_json()
    parsed = AgentConfig.model_validate_json(json_str)
    assert parsed == cfg


def test_invalid_llm_temperature_raises():
    with pytest.raises(ValueError):
        LLMConfig(model_name="gpt-4", temperature=3.0)


def test_agent_memory_roundtrip():
    msg1 = Message(role="user", content="Hi")
    msg2 = Message(role="assistant", content="Hello")
    memory = AgentMemory(messages=[msg1, msg2])
    json_str = memory.model_dump_json()
    parsed = AgentMemory.model_validate_json(json_str)
    assert parsed == memory


def test_agent_memory_requires_non_empty():
    with pytest.raises(ValueError):
        AgentMemory(messages=[])


def test_agent_workflow_input_roundtrip():
    session_id = str(uuid4())
    awi = AgentWorkflowInput(session_id=session_id, user_message="Hello", run_id=None)
    json_str = awi.model_dump_json()
    parsed = AgentWorkflowInput.model_validate_json(json_str)
    assert parsed == awi


def test_agent_workflow_output_roundtrip():
    run_id = str(uuid4())
    final_msg = Message(role="assistant", content="Done")
    awo = AgentWorkflowOutput(run_id=run_id, status="completed", final_message=final_msg)
    json_str = awo.model_dump_json()
    parsed = AgentWorkflowOutput.model_validate_json(json_str)
    assert parsed == awo 
