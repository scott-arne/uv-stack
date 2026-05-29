from uvstack.errors import (
    ConfigError,
    EnvError,
    ResolutionError,
    ToolError,
    UvstackError,
)


def test_uvstack_error_carries_message_and_hint():
    err = UvstackError("boom", hint="try this")
    assert err.message == "boom"
    assert err.hint == "try this"
    assert str(err) == "boom"


def test_subclasses_are_uvstack_errors():
    assert issubclass(ConfigError, UvstackError)
    assert issubclass(ResolutionError, UvstackError)
    assert issubclass(EnvError, UvstackError)
    assert issubclass(ToolError, UvstackError)


def test_tool_error_records_command_and_returncode():
    err = ToolError("compile failed", command=["uv", "pip", "compile"], returncode=2)
    assert err.command == ["uv", "pip", "compile"]
    assert err.returncode == 2
    assert err.message == "compile failed"
