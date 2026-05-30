from uv_stack.errors import (
    ConfigError,
    EnvError,
    ResolutionError,
    ToolError,
    UvStackError,
)


def test_uv_stack_error_carries_message_and_hint():
    err = UvStackError("boom", hint="try this")
    assert err.message == "boom"
    assert err.hint == "try this"
    assert str(err) == "boom"


def test_subclasses_are_uv_stack_errors():
    assert issubclass(ConfigError, UvStackError)
    assert issubclass(ResolutionError, UvStackError)
    assert issubclass(EnvError, UvStackError)
    assert issubclass(ToolError, UvStackError)


def test_tool_error_records_command_and_returncode():
    err = ToolError("compile failed", command=["uv", "pip", "compile"], returncode=2)
    assert err.command == ["uv", "pip", "compile"]
    assert err.returncode == 2
    assert err.message == "compile failed"
