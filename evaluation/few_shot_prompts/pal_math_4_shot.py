from .few_shot_prompting import FewShotPrompting

few_shot_prompt = """Problem:
Find the value of $x$ that satisfies $\\frac{\\sqrt{3x+5}}{\\sqrt{6x+5}}=\\frac{\\sqrt{5}}{3}$. Express your answer as a common fraction.

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Initialize x
x = symbols('x')

# Define the equation
equation = Eq(sqrt(3*x + 5)/sqrt(6*x + 5), sqrt(5)/3)

# Solve for x
answer = solve(equation, x)
```
The imports required for this program are
```
from sympy import symbols, Eq, solve, sqrt
```
I hope my solution is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Given det(A) = 2 and det(B) = 12
det_A = 2
det_B = 12

# Use the property det(AB) = det(A)*det(B)
det_AB = det_A * det_B

answer = det_AB
```
The imports required for this program are
```

```
I hope my solution is correct. 

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Calculate the total weight lifted initially, which is 2*20*12 pounds
total_weight = 2 * 20 * 12

# Since Terrell lifts two 15-pound weights, divide the total weight by 2 * 15
repetitions = total_weight / (2*15)

answer = n_value 
```
The imports required for this program are
```

```
I hope my solution is correct. 

Problem:
If Anna flips 8 coins, what is the probability that she gets more heads than tails?

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# There are 2**8 possible outcomes
n = 8
total_outcomes = 2 ** n

# There are binom(n, k) ways to get k heads
favorable_outcomes = 0
for k in range((n // 2) + 1, n + 1):
    favorable_outcomes += math.comb(n, k)
    
probability = favorable_outcomes / total_outcomes

answer = probability
```
The imports required for this program are
```
import math
```
I hope my solution is correct.

Problem:
Evaluate $\\left\\lceil3\\left(6-\\frac12\\right)\\right\\rceil$.

You are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.
```
# Calculate 3 * (6 - 1/2)
result = 3 * (6 - 0.5)

# Apply the ceiling function
ceiling_result = math.ceil(result)

answer = ceiling_result
```
The imports required for this program are
```
import math
```
I hope my solution is correct."""

class PALMathPrompt(FewShotPrompting):
    def __init__(self):
        super().__init__()

    def format_prompt(self, task_input, task_output):
        prompt = f"{few_shot_prompt}\n\nProblem:\n{task_input}\n\nYou are an expert programmer. Solve the above mathematical problem by writing a Python program. Express your answer as a numeric type or a SymPy object.\n{task_output}"
        return prompt.rstrip()

    def stop_words(self):
        return ["\nProblem:", "Problem:"]
