import ast


class CodeValidationError(Exception):
    """Custom exception for code validation errors."""


class EmptyCodeError(CodeValidationError):
    """Raised when the generated code is empty."""

    def __init__(self, message: str = "Generated code is empty."):
        super().__init__(message)


class SyntaxErrorInCode(CodeValidationError):
    """Raised when the generated code contains syntax errors."""

    def __init__(self, error: SyntaxError):
        super().__init__(f"Syntax error in generated code: {error}")


class MissingFunctionError(CodeValidationError):
    """Raised when the generated code is missing any functions."""

    def __init__(self, function_name: str):
        super().__init__(
            f"Generated code is missing required function: {function_name}"
        )


def validate_code(code: str):
    """
    Validates that the given code is syntactically correct and contains at least one function definition.

    Args:
        code (str): The code to validate.
    Raises:
        EmptyCodeError: If the code is empty or only whitespace.
        SyntaxErrorInCode: If the code contains syntax errors.
        MissingFunctionError: If the code does not contain any function definitions.
    """
    if not code or not code.strip():
        raise EmptyCodeError("Generated code is empty or whitespace.")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxErrorInCode(e) from e

    if not any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)):
        raise MissingFunctionError("At least one function definition is required.")
