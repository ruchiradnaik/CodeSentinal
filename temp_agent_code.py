# Verified by CodeSentinel
def divide_numbers(a, b):
    return a / b

if __name__ == "__main__":
    try:
        result = divide_numbers(10, 0)
        print(f"Result: {result}")
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")