# Module 1.2: Python for AI/ML - Prerequisites

## Required Prior Knowledge

Before starting this module, verify you can complete these tasks:

### From Module 1.1: DGX Spark Platform Mastery

| Skill | Self-Check | Reference |
|-------|------------|-----------|
| Start NGC container | Can you run `docker run --gpus all ...` successfully? | Module 1.1, Lab 1 |
| Verify GPU access | Does `nvidia-smi` show the GPU inside the container? | Module 1.1 |
| Launch Jupyter Lab | Can you access JupyterLab at `localhost:8888`? | Module 1.1 |
| Navigate filesystem | Can you find `/workspace` inside the container? | Module 1.1 |

### Python Fundamentals (External Prerequisite)

| Skill | Self-Check |
|-------|------------|
| Variables and types | Can you create variables of type `int`, `float`, `str`, `list`, `dict`? |
| Control flow | Can you write `if/else` statements and `for`/`while` loops? |
| Functions | Can you define functions with parameters and return values? |
| Classes | Can you create a simple class with `__init__` and methods? |
| List comprehensions | Can you write `[x**2 for x in range(10)]`? |
| File I/O | Can you read and write text files with `open()`? |

---

## Quick Self-Assessment

Run this code in Python. If all assertions pass, you're ready for Module 1.2:

```python
# Test 1: Basic Python
def test_python_basics():
    # Variables and types
    x = 42
    name = "DGX Spark"
    items = [1, 2, 3]
    config = {"gpu": True, "memory": 128}

    assert isinstance(x, int)
    assert isinstance(name, str)
    assert isinstance(items, list)
    assert isinstance(config, dict)

    # List comprehension
    squares = [i**2 for i in range(5)]
    assert squares == [0, 1, 4, 9, 16]

    # Dictionary access
    assert config["memory"] == 128

    print("✅ Python basics: PASSED")

# Test 2: Functions and classes
def test_functions_classes():
    # Function with default parameter
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    assert greet("Spark") == "Hello, Spark!"
    assert greet("Spark", "Hi") == "Hi, Spark!"

    # Simple class
    class Counter:
        def __init__(self, start=0):
            self.value = start

        def increment(self):
            self.value += 1
            return self.value

    c = Counter(10)
    assert c.increment() == 11

    print("✅ Functions and classes: PASSED")

# Run tests
test_python_basics()
test_functions_classes()
print("\n🎉 All prerequisites satisfied! You're ready for Module 1.2.")
```

---

## Skill Check Answers

<details>
<summary>Can you create variables of type int, float, str, list, dict?</summary>

```python
my_int = 42
my_float = 3.14
my_str = "hello"
my_list = [1, 2, 3]
my_dict = {"key": "value"}
```

</details>

<details>
<summary>Can you write if/else statements and loops?</summary>

```python
# If/else
x = 10
if x > 5:
    print("Greater")
else:
    print("Less or equal")

# For loop
for i in range(3):
    print(i)

# While loop
count = 0
while count < 3:
    print(count)
    count += 1
```

</details>

<details>
<summary>Can you define functions with parameters and return values?</summary>

```python
def add(a, b):
    return a + b

def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

result = add(2, 3)  # 5
message = greet("World")  # "Hello, World!"
```

</details>

<details>
<summary>Can you create a simple class?</summary>

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

rect = Rectangle(4, 5)
print(rect.area())       # 20
print(rect.perimeter())  # 18
```

</details>

---

## What You'll Learn in This Module

Building on your Python fundamentals, you'll learn:

1. **NumPy** - Efficient array operations (100x faster than Python loops)
2. **Pandas** - Data manipulation and preprocessing
3. **Matplotlib/Seaborn** - Publication-quality visualizations
4. **Einsum** - Compact tensor notation used in transformers
5. **Profiling** - Finding and fixing performance bottlenecks

---

## Estimated Time Investment

| Your Background | Estimated Module Time |
|-----------------|----------------------|
| Strong Python, some NumPy | ~6 hours |
| Comfortable Python, new to NumPy | ~10 hours |
| Basic Python, no scientific computing | ~14 hours |

---

## Need to Brush Up?

If you need to review Python fundamentals:

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python - Python Basics](https://realpython.com/python-first-steps/)
- [Codecademy Python Course](https://www.codecademy.com/learn/learn-python-3)

---

## Next Steps

Once you've verified all prerequisites:

1. Complete the [QUICKSTART.md](./QUICKSTART.md) (5 minutes)
2. Read the [STUDY_GUIDE.md](./STUDY_GUIDE.md) for the learning roadmap
3. Start with Lab 1.2.1: NumPy Broadcasting
