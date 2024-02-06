from .few_shot_prompting import FewShotPrompting

few_shot_prompt = """
问题 1.   设数列 $\\left\\{a_{n}\\right\\}$ 的前 $n$ 项和为 $S_{n}$, 且 $a_{1}=-1, a_{n+1}=S_{n+1} S_{n}$, 则 $S_{n}=(\\quad)$.
问题 1的解析:   让我们写出这个数列的前n项和：
$S_n = a_1 + a_2 + ... + a_n$
$S_n = -1 + (S_2 S_1) + (S_3 S_2) + ... + (S_{n+1} S_n)$
$S_n = -1 + (S_2 S_1) + (S_3 S_2) + ... + (S_n S_{n-1}) + (S_{n+1} S_n)$
$S_n = -1 + S_n (S_{n+1} - S_1)$
$S_n - S_n S_{n+1} = -1 - S_n$
$S_n (1 - S_{n+1}) = -1 - S_n$
$S_n = -\\frac{1}{1 - S_{n+1}}$
因为这个数列后面的所有项都是0，我们可以看到对于所有$n\\geq 1$，$S_{n+1} = 0$。因此，我们有：
$S_n = -\\frac{1}{1 - S_{n+1}} = -\\frac{1}{1 - 0} = -1$
这个数列前n项和的公式是$S_n = -\\frac{1}{n}$。
答案是 $-\\frac{1}{n}$


问题 2.   若 $\\left(x+\\frac{1}{x}\\right)^{n}$ 的展开式中第 3 项与第 7 项的二项式系数相等, 则该展 开式中 $\\frac{1}{x^{2}}$ 的系数为 $(\\quad)$.
问题 2的解析:   由题意可得, $c_{n}^{2}=c_{n}^{6}$
$\\therefore n=8$
展开式的通项 T_{r+1}=C_8^r x^{8-r}\\left(\\frac{1}{x}\\right)^r=C_8^r x^{8-2 r}$
令 $8-2 r=-2$ 可得 $r=5$
此时系数为 $c_{8}^{5}=56$
答案是 56


问题 3.   函数 $\\mathrm{f}(\\mathrm{x})=\\sin (\\mathrm{x}+2 \\phi)-2 \\sin \\phi \\cos (\\mathrm{x}+\\phi)$ 的最大值为 $(\\quad)$.
问题 3的解析:   函数 $f(x)=\\sin (x+2 \\phi)-2 \\sin \\phi \\cos (x+\\phi)=\\sin [(x+\\phi)+\\phi]-$ $2 \\sin \\phi \\cos (x+\\phi)$
$=\\sin (x+\\phi) \\cos \\phi+\\cos (x+\\phi) \\sin \\phi-2 \\sin \\phi \\cos (x+\\phi)=\\sin (x+\\phi) \\cos \\phi-\\cos$ $(x+\\phi) \\sin \\phi$ $=\\sin [(x+\\phi)-\\phi]=\\sin x$
故函数 $f(x)$ 的最大值为 1
答案是 1


问题 4.   已知向量 $\\vec{a}=(3,1), \\vec{b}=(1,0), \\vec{c}=\\vec{a}+k \\vec{b}$. 若 $\\vec{a} \\perp \\vec{c}$, 则 $k=(\\quad)$
问题 4的解析:   \\because \\vec{a}=(3,1), \\vec{b}=(1,0), \\therefore \\vec{c}=\\vec{a}+k \\vec{b}=(3+k, 1)$ ，
$\\because \\vec{a} \\perp \\vec{c}, \\therefore \\vec{a} \\square \\vec{c}=3(3+k)+1 \\times 1=0$, 解得 $k=-\\frac{10}{3}$
答案是 $-\\frac{10}{3}$


问题 5.   设向量 $\\vec{a}, \\vec{b}$ 不平行, 向量 $\\lambda \\vec{a}+\\vec{b}$ 与 $\\vec{a}+2 \\vec{b}$ 平行, 则实数 $\\lambda=(\\quad)$.
问题 5的解析:   $\\because$ 向量 $\\vec{a}, \\vec{b}$ 不平行, 向量 $\\lambda \\vec{a}+\\vec{b}$ 与 $\\vec{a}+2 \\vec{b}$ 平行,
$\\therefore \\lambda \\vec{a}+\\vec{b}=t(\\vec{a}+2 \\vec{b})=t \\vec{a}+2 t \\vec{b}$
$\\therefore\\left\\{\\begin{array}{c}\\lambda=\\mathrm{t} \\\\ 1=2 \\mathrm{t},\\end{array}\\right.$ 解得实数 $\\lambda=\\frac{1}{2}$.
答案是 $\\frac{1}{2}$
""".strip()

class CoTGaoKaoMathClozePrompt(FewShotPrompting):
    def __init__(self):
        super().__init__()

    def format_prompt(self, task_input, task_output):
        prompt = f"{few_shot_prompt}\n\n\n问题 6.   {task_input}\n问题 6的解析:   {task_output}"
        return prompt.rstrip()

    def stop_words(self):
        return ["\n问题 "]
