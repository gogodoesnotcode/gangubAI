"""Arithmetic calculator tool."""

from langchain_core.tools import tool


@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Perform basic arithmetic operations.

    Use this tool when you need to calculate: addition, subtraction, multiplication, or division.

    Args:
        first_num: The first number in the calculation
        second_num: The second number in the calculation
        operation: Must be one of: "add", "subtract", "multiply", "divide"

    Returns:
        A string with the calculation result
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiply":
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return "Error: Cannot divide by zero"
            result = first_num / second_num
        else:
            return f"Error: Invalid operation '{operation}'. Use: add, subtract, multiply, or divide"
        return f"The result of {first_num} {operation} {second_num} is {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"
