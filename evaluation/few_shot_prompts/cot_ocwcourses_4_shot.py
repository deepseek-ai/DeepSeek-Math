from .few_shot_prompting import FewShotPrompting

few_shot_prompt = """
Problem:                                                                                
Subproblem 0: What is the net charge of arginine in a solution of $\\mathrm{pH} 1.0$? 
Please format your answer as +n or -n.                           
Solution:
The answer is +2.                                                     
Final answer: The final answer is $\\boxed{+2}$. I hope it is correct.

Problem:
Subproblem 0: Let $z = 1 + \\sqrt{3} i$. Find $a, b$ that satisfy the equation 
$z^4 = a + bi$. Express your answer as the ordered pair $(a,b)$.         
Solution:
$z^{4}$ has argument $4 \\pi / 3$ and radius 16 , so it's equal to $-8-8 \\sqrt{3} i$. 
Thus $a = -8, b = -8\\sqrt 3$, and our answer is $(-8, -8\\sqrt{3})$.
Final answer: The final answer is $\\boxed{(-8, -8\\sqrt{3})}$. I hope it is correct.

Problem:
Preamble: For each Laplace Transform \\(Y(s)\\), find the function \\(y(t)\\):
Subproblem 0: 
\\[Y(s)=\\frac{1}{(s+a)(s+b)}\\]
Solution:
We can simplify with partial fractions:
\\[Y(s)=\\frac{1}{(s+a)(s+b)}=\\frac{C}{s+a}+\\frac{D}{s+b}\\]
find the constants 
\\(C\\) and \\(D\\) by setting \\(s=-a\\) and \\(s=-b\\)
\\[
  \\begin{aligned}
  \\frac{1}{(s+a)(s+b)} &=\\frac{C}{s+a}+\\frac{D}{s+b} \\\\
  1 &=C(s+b)+D(s+a) \\\\
  C &=\\frac{1}{b-a} \\\\
  D &=\\frac{1}{a-b}
  \\end{aligned}
\\]
therefore
\\[
Y(s)=\\frac{1}{b-a} \\frac{1}{s+a}-\\frac{1}{b-a} \\frac{1}{s+b}
\\]
By looking up the inverse Laplace Transform of \\(\\frac{1}{s+b}\\), we find the total 
solution \\(y(t)\\)
\\[
  y(t)=\\frac{e^{-a t}-e^{-b t}}{b-a}
\\].
Final answer: The final answer is $\\boxed{\\frac{e^{-a t}-e^{-b t}}{b-a}}$. I hope it is correct.

Problem:
Preamble: The following subproblems refer to the differential equation 
$\\ddot{x}+b \\dot{x}+x=0$.
Subproblem 0: What is the characteristic polynomial $p(s)$ of 
$\\ddot{x}+b \\dot{x}+x=0$?
Solution:
The characteristic polynomial is $p(s)=s^{2}+b s+1$.
Final answer: The final answer is $\\boxed{s^{2}+b s+1}$. I hope it is correct.
""".strip()

few_shot_prompt = """
Problem:                                                                                
Subproblem 0: What is the net charge of arginine in a solution of $\\mathrm{pH} 1.0$? 
Please format your answer as +n or -n.                           
Solution:
The answer is +2.                                                     
Final answer: The final answer is +2. I hope it is correct.

Problem:
Subproblem 0: Let $z = 1 + \\sqrt{3} i$. Find $a, b$ that satisfy the equation 
$z^4 = a + bi$. Express your answer as the ordered pair $(a,b)$.         
Solution:
$z^{4}$ has argument $4 \\pi / 3$ and radius 16 , so it's equal to $-8-8 \\sqrt{3} i$. 
Thus $a = -8, b = -8\\sqrt 3$, and our answer is $\\boxed{(-8, -8\\sqrt{3})}$.
Final answer: The final answer is (-8, -8\\sqrt{3}). I hope it is correct.

Problem:
Preamble: For each Laplace Transform \\(Y(s)\\), find the function \\(y(t)\\):
Subproblem 0: 
\\[Y(s)=\\boxed{\\frac{1}{(s+a)(s+b)}}\\]
Solution:
We can simplify with partial fractions:
\\[Y(s)=\\frac{1}{(s+a)(s+b)}=\\frac{C}{s+a}+\\frac{D}{s+b}\\]
find the constants 
\\(C\\) and \\(D\\) by setting \\(s=-a\\) and \\(s=-b\\)
\\[
  \\begin{aligned}
  \\frac{1}{(s+a)(s+b)} &=\\frac{C}{s+a}+\\frac{D}{s+b} \\\\
  1 &=C(s+b)+D(s+a) \\\\
  C &=\\frac{1}{b-a} \\\\
  D &=\\frac{1}{a-b}
  \\end{aligned}
\\]
therefore
\\[
Y(s)=\\frac{1}{b-a} \\frac{1}{s+a}-\\frac{1}{b-a} \\frac{1}{s+b}
\\]
By looking up the inverse Laplace Transform of \\(\\frac{1}{s+b}\\), we find the total 
solution \\(y(t)\\)
\\[
  y(t)=\\boxed{\\frac{1}{b-a}\\left(e^{-a t}-e^{-b t}\\right)}
\\].
Final answer: The final answer is \\[\\frac{1}{b-a}\\left(e^{-a t}-e^{-b t}\\right)\\]. I hope it is correct.

Problem:
Preamble: The following subproblems refer to the differential equation 
$\\ddot{x}+b \\dot{x}+x=0$.
Subproblem 0: What is the characteristic polynomial $p(s)$ of 
$\\ddot{x}+b \\dot{x}+x=0$?
Solution:
The characteristic polynomial is $p(s)=\\boxed{s^{2}+b s+1}$.
Final answer: The final answer is $s^{2}+b s+1$. I hope it is correct.
""".strip()

class OCWCoursesPrompt(FewShotPrompting):
    def __init__(self):
        super().__init__()

    def format_prompt(self, task_input, task_output):
        prompt = f"{few_shot_prompt}\n\nProblem:\n{task_input}\nSolution:\n{task_output}"
        return prompt.rstrip()

    def stop_words(self):
        return ["\nProblem:"]
