from .few_shot_prompting import FewShotPrompting

few_shot_prompt = {
    'numbertheory': """Informal:
(*### Problem

Find the minimum value of $\\frac{9x^2\\sin^2 x + 4}{x\\sin x}$ for $0 < x < \\pi$. Show that it is 12.

### Solution

Let $y = x \\sin x$. It suffices to show that $12 \\leq \\frac{9y^2 + 4}{y}.
It is trivial to see that $y > 0$. 
Then one can multiply both sides by $y$ and it suffices to show $12y \\leq 9y^2 + 4$.
This can be done by the sum of squares method.*)

Formal:
theorem aime_1983_p9:
  fixes x::real
  assumes "0<x" "x<pi"
  shows "12 \\<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)"
proof -
  (* Let $y = x \\sin x$. *)
  define y where "y=x * sin x"
  (* It suffices to show that $12 \\leq \\frac{9y^2 + 4}{y}. *)
  have "12 \\<le> (9 * y^2 + 4) / y"
  proof -
    (* It is trivial to see that $y > 0$. *)
    have c0: "y > 0"
      sledgehammer
    (* Then one can multiply both sides by $y$ and it suffices to show $12y \\leq 9y^2 + 4$. *)
    have "(9 * y^2 + 4) \\<ge> 12 * y" 
      sledgehammer
    then show ?thesis
      sledgehammer
  qed
  then show ?thesis
    sledgehammer
qed



Informal:
(*### Problem

Find the greatest common factor of 180 and 168. Show that it is 12.

### Solution

This is true by simple evaluation.*)

Formal:
theorem mathd_numbertheory_188:
  "gcd 180 168 = (12::nat)"
  sledgehammer



Informal:
(*### Problem

Show that for positive integer n, 2 divides $4^n$.

### Solution

Since n is positive, we can find a natural number m where $m+1=n$.
Then we can show that 2 divides $4^{m+1}$. The conclusion thus follows.*)

Formal:
theorem numbertheory_2dvd4expn:
  fixes n :: nat
  assumes h0 : "n \\<noteq> 0"
  shows "(2::nat) dvd 4^n"
proof -
  obtain m::nat where c0: "m+1=n"
    sledgehammer
  have "(2::nat) dvd 4^(m+1)" sledgehammer
  then show ?thesis unfolding c0 sledgehammer
qed



Informal:
(*### Problem

What is the remainder when $1 + 2 + 3 + 4 + \\dots + 9 + 10$ is divided by 9? Show that it is 1.

### Solution

This is true by simple evaluation.*)

Formal:
theorem mathd_numbertheory_466:
  "(\\<Sum> k< 11. k) mod 9 = (1::nat)"
  sledgehammer



Informal:
(*### Problem

If $321_{b}$ is equal to the base 10 integer 57, find $b$ given that $b>0$. Show that it is 4.

### Solution

Converting $321_{b}$ to base 10 and setting it equal to 57, we find that  \\begin{align*} 3(b^2)+2(b^1)+1(b^0)&=57
\\\\ 3b^2+2b+1&=57
\\\\\\Rightarrow\\qquad 3b^2+2b-56&=0
\\\\\\Rightarrow\\qquad (3b+14)(b-4)&=0
\\end{align*}This tells us that $b$ is either $-\\frac{14}{3}$ or $4$. We know that $b>0$, so $b=4$.*)

Formal:
theorem mathd_numbertheory_48:
  fixes b :: real
  assumes h0 : "0<b"
    and h1 : "3 * b^2 + 2 * b + 1 = 57"
  shows "b=4"
proof -
  (* Converting $321_{b}$ to base 10 and setting it equal to 57, we find that  \\begin{align*} 3(b^2)+2(b^1)+1(b^0)&=57
  \\\\ 3b^2+2b+1&=57
  \\\\\\Rightarrow\\qquad 3b^2+2b-56&=0
  \\\\\\Rightarrow\\qquad (3b+14)(b-4)&=0
  \\end{align*} *)
  have "0 = 3 * b^2 + 2 * b -56" using h1 sledgehammer
  also have "... = (3*b+14)*(b-4)" sledgehammer
  finally have "0 = (3*b+14)*(b-4)" sledgehammer
  (* This tells us that $b$ is either $-\\frac{14}{3}$ or $4$. *)
  then have "b = -14/3 ∨ b=4" sledgehammer
  (* We know that $b>0$, so $b=4$. *)
  then show ?thesis using h0 sledgehammer
qed



Informal:
(*### Problem

When Rachel divides her favorite number by 7, she gets a remainder of 5. What will the remainder be if she multiplies her favorite number by 5 and then divides by 7? Show that it is 4.

### Solution

Let $n$ be Rachel's favorite number. 
Then $n \\equiv 5 \\pmod{7}$, so $5n \\equiv 5 \\cdot 5 \\equiv 25 \\equiv 4 \\pmod{7}$.
*)

Formal:
theorem mathd_numbertheory_335:
  fixes n :: nat
  assumes h0 : "n mod 7 = 5"
  shows "(5 * n) mod 7 = 4"
proof -
  (* Then $n \\equiv 5 \\pmod{7}$, so $5n \\equiv 5 \\cdot 5 \\equiv 25 \\equiv 4 \\pmod{7}$. *)
  have c0:"(5 * n) mod 7 = (5 * 5) mod 7" using h0
    sledgehammer
  then have "\\<dots> = 4" sledgehammer
  then have "(5 * n) mod 7 = 4" using c0 sledgehammer
  then show ?thesis sledgehammer
qed



Informal:
(*### Problem

What positive two-digit integer is exactly twice the sum of its digits? Show that it is 18.

### Solution

We simplify $10a + b = 2(a+b)$ to get $8a = b$.
Since $a$ is at least 1, $b$ is at least 8.
We know $b$ is 8 since $8a = b$ and $a$ is a natural number.
Hence $a$ is 1.
The two-digit integer is hence $18$.
*)

Formal:
theorem mathd_numbertheory_284:
  fixes a b :: nat
  assumes h0 : "1\\<le>a \\<and> a \\<le>9 \\<and> b \\<le>9"
    and h1 : "10 * a + b = 2 * (a+b)"
  shows "10 * a + b = 18"
proof -
  (* We simplify $10a + b = 2(a+b)$ to get $8a = b$. *)
  have c0: "8 * a = b" using h1 sledgehammer
  (* Since $a$ is at least 1, $b$ is at least 8. *)
  hence "b \\<ge> 8" using h0 sledgehammer
  (* We know $b$ is 8 since $8a = b$ and $a$ is a natural number. *)
  hence c1:"b = 8" using h0 c0
    sledgehammer
  (* Hence $a$ is 1. *)
  hence "a = 1" using c0 sledgehammer
  (* The two-digit integer is hence $18$. *)
  then show ?thesis using c1 sledgehammer
qed



""".strip(),
    "other": """Informal:
(*### Problem

Find the minimum value of $\\frac{9x^2\\sin^2 x + 4}{x\\sin x}$ for $0 < x < \\pi$. Show that it is 12.

### Solution

Let $y = x \\sin x$. It suffices to show that $12 \\leq \\frac{9y^2 + 4}{y}.
It is trivial to see that $y > 0$. 
Then one can multiply both sides by $y$ and it suffices to show $12y \\leq 9y^2 + 4$.
This can be done by the sum of squares method.*)

Formal:
theorem aime_1983_p9:
  fixes x::real
  assumes "0<x" "x<pi"
  shows "12 \\<le> ((9 * (x^2 * (sin x)^2)) + 4) / (x * sin x)"
proof -
  (* Let $y = x \\sin x$. *)
  define y where "y=x * sin x"
  (* It suffices to show that $12 \\leq \\frac{9y^2 + 4}{y}. *)
  have "12 \\<le> (9 * y^2 + 4) / y"
  proof -
    (* It is trivial to see that $y > 0$. *)
    have c0: "y > 0"
      sledgehammer
    (* Then one can multiply both sides by $y$ and it suffices to show $12y \\leq 9y^2 + 4$. *)
    have "(9 * y^2 + 4) \\<ge> 12 * y" 
      sledgehammer
    then show ?thesis
      sledgehammer
  qed
  then show ?thesis
    sledgehammer
qed



Informal:
(*### Problem

Show that for any four complex numbers a, b, c, and d, $(a-d)(a-c)(a-b) = -(((a^2 - a(b+c)) + bc) * d) + (a^2 - a(b+c) + bc) * a$.

### Solution

We first see that $a^2 = a * a$ trivially.
Unfolding this, the main equation holds true when terms are rearranged.*)

Formal:
theorem algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta:
  fixes a b c d :: complex
  shows "(a-d) * (a-c) * (a-b) = -(((a^2 - (b+c) * a) + c * b) * d) + (a^2 - (b+c) * a + c * b) * a"
proof -
  (* We first see that $a^2 = a * a$ trivially. *)
  have t0: "a^2 = a * a"
    using power2_eq_square
      sledgehammer
  (* Unfolding this, the main equation holds true when terms are rearranged. *)
  show ?thesis unfolding t0
    sledgehammer
qed



Informal:
(*### Problem

Find the greatest common factor of 180 and 168. Show that it is 12.

### Solution

This is true by simple evaluation.*)

Formal:
theorem mathd_numbertheory_188:
  "gcd 180 168 = (12::nat)"
  sledgehammer



Informal:
(*### Problem

For how many positive integers $n$ is $n^2 - 3n + 2$ a [[prime]] number?

$\\mathrm{(A)}\\ \\text{none}
\\qquad\\mathrm{(B)}\\ \\text{one}
\\qquad\\mathrm{(C)}\\ \\text{two}
\\qquad\\mathrm{(D)}\\ \\text{more\\ than\\ two,\\ but\\ finitely\\ many}
\\qquad\\mathrm{(E)}\\ \\text{infinitely\\ many}$ Show that it is \\mathrm{(B)}\\ \\text{one}.

### Solution

Factoring, we get $n^2 - 3n + 2 = (n-2)(n-1)$. 
Either $n-1$ or $n-2$ is odd, and the other is even.  
Their product must yield an even number.  
The only prime that is even is $2$, which is when $n$ is $3$ or $0$. 
Since $0$ is not a positive number, the answer is $\\mathrm{(B)}\\ \\text{one}$.*)

Formal:
theorem amc12b_2002_p3:
  fixes n ::nat
  assumes "n>0"
    and prime:"prime (n^2+2-3*n)"
  shows "n=3"
proof -
  have "n>2" 
  proof (rule ccontr)
    assume "\\<not> 2 < n"
    then have "n=1 \\<or> n=2" using \\<open>n>0\\<close> sledgehammer
    then show False using prime[THEN prime_gt_1_nat]
      sledgehammer
  qed
  (* Factoring, we get $n^2 - 3n + 2 = (n-2)(n-1)$. *)
  then have "n^2+2-3*n  = (n-1) * (n-2)"
    unfolding power2_eq_square
    sledgehammer
  (* Either $n-1$ or $n-2$ is odd, and the other is even.  
  Their product must yield an even number.  
  The only prime that is even is $2$, which is when $n$ is $3$ or $0$. 
  Since $0$ is not a positive number, the answer is $\\mathrm{(B)}\\ \\text{one}$.*)
  then have "prime ((n-1) * (n-2))"
    using prime sledgehammer
  then have "n-1=1 \\<or> n-2 = 1"
    using prime_product sledgehammer
  with \\<open>n>2\\<close>
  show "n=3" sledgehammer
qed



Informal:
(*### Problem

For a positive real number a, show that $10a\\leq 28a^2+1$.

### Solution

It suffices to show $0\\leq 28a^2 - 10a + 1$.
First, consider completing the square for $28a^2 - 10a$ and observe that $(a - \\frac{5}{28})^2 = a^2 - \\frac{10}{28}a + (5/28)^2$.
Since $0\\leq (a - \\frac{5}{28})^2$, we have $0\\leq a^2 - \\frac{10}{28}a + (5/28)^2$.
Multiplying by 28 and simplifying terms gives $0\\leq 28*a^2 - 10*a + (25/28)$.
Since $25/28 < 1$, the result follows.*)

Formal:
theorem algebra_binomnegdiscrineq_10alt28asqp1:
  fixes a :: real
  shows "10 * a \\<le> 28 * a^2 + 1"
proof -
(* it suffices to show $0\\leq 28a^2 - 10a + 1$ *)
  have c0: "0 \\<le> 28*a^2 - 10*a + 1"
  proof -
    (* observe that $(a - \\frac{5}{28})^2 = a^2 - \\frac{10}{28}a + (5/28)^2$ *)
    have c1: "(a - (5/28))^2 = a^2 - 10/28*a + (5/28)^2"
      sledgehammer
    (* we have $0\\leq a^2 - \\frac{10}{28}a + (5/28)^2$ *)
    then have c2: "0 \\<le> a^2 - 10/28*a + (5/28)^2" using c1
      sledgehammer
    (* Multiplying by 28 and simplifying terms gives $0\\leq 28*a^2 - 10*a + (25/28)$ *)
    then have c3: "0 \\<le> 28*a^2 - 10*a + 28*((5/28)^2)" using c2
      sledgehammer
    then have c4: "0 \\<le> 28*a^2 - 10*a + 28*((5/28)*(5/28))" using c3
      sledgehammer
    then have c5: "0 \\<le> 28*a^2 - 10*a + (25/28)" using c4
      sledgehammer
    (* Since $25/28 < 1$, the result follows. *)
    then show ?thesis using c5
      sledgehammer
  qed
  then show ?thesis
    sledgehammer
qed



Informal:
(*### Problem

Show that for any complex number a, $(a-10)(a+11) = a^2 + a - 110$.

### Solution

We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$.
This equals $a^2 + a - 10*11 = a^2 + a - 110$.*)

Formal:
theorem algebra_2rootsintpoly_am10tap11eqasqpam110:
  fixes a :: complex
  shows "(a-10) * (a+11) = a^2 + a -110"
proof -
  (* We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$. *)
  have "(a-10) * (a+11) = a^2 - 10*a + 11*a - 10 *11"
    sledgehammer
  (* This equals $a^2 + a - 10*11 = a^2 + a - 110$. *)
  also have "\\<dots> = a^2 + a - 10 * 11"
    sledgehammer
  also have "\\<dots> = a^2 + a - 110"
    sledgehammer
  finally show ?thesis
    sledgehammer
qed



""".strip()
}

class MiniF2FIsabellePrompt(FewShotPrompting):
    def __init__(self):
        super().__init__()

    def format_prompt(self, task_input, task_output):
        if 'numbertheory' in task_input.split("Formal:", 1)[1]:
            tag = 'numbertheory'
        else:
            tag = 'other'
        prompt = f"{few_shot_prompt[tag].strip()}\n\n\n\nInformal:\n{task_input.strip()}\n{task_output.strip()}"
        return prompt.rstrip()

    def stop_words(self):
        return ["\nInformal:"]
