import regex

from data_processing.answer_extraction import extract_math_answer, strip_string

def process_gsm8k_test(item):
    sample = {
        'dataset': 'gsm8k-cot',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': item['question']},
            {'role': 'assistant', 'content': regex.sub(r"<<[^<>]*>>", "", item['cot']) + "\nSo the answer is $\\boxed{" + item['answer'].strip() + "}$."}
        ],
        'answer': item['answer'].replace(',', '')
    }
    yield sample

def process_math_test(item):
    question = item["problem"]
    try:
        answer = extract_math_answer(question, item['solution'], task="cot")
    except:
        return
    sample = {
        "dataset": "math-cot",
        "id": item['id'],
        "level": item["level"],
        "type": item["type"],
        "category": item["category"],
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n".join(regex.split(r"(?<=\.) (?=[A-Z])", item["solution"]))}
        ],
        "answer": answer
    }
    yield sample

def process_math_sat(item):
    options = item['options'].strip()
    assert 'A' == options[0]
    options = '(' + options
    for ch in 'BCDEFG':
        if f' {ch}) ' in options:
            options = regex.sub(f' {ch}\) ', f" ({ch}) ", options)
    question = f"{item['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': item['Answer']}
    ]
    item = {
        'dataset': 'math_sat',
        'id': item['id'],
        'language': 'en',
        'messages': messages,
        'answer': item['Answer'],
    }
    yield item

def process_ocwcourses(item):
    messages = [
        {'role': 'user', 'content': item['problem'].strip()},
        {'role': 'assistant', 'content': item['solution'].strip()}
    ]
    item = {
        "dataset": "OCWCourses",
        "id": item['id'],
        "language": "en",
        "messages": messages,
        "answer": item['answer']
    }
    yield item

def process_mmlu_stem(item):
    options = item['options']
    for i, (label, option) in enumerate(zip('ABCD', options)):
        options[i] = f"({label}) {str(option).strip()}"
    options = ", ".join(options)
    question = f"{item['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': item['answer']}
    ]
    item = {
        "dataset": "MMLU-STEM",
        "id": item['id'],
        "language": "en",
        "messages": messages,
        "answer": item['answer']
    }
    yield item

def process_mgsm_zh(item):
    item['answer'] = item['answer'].replace(',', '')
    yield item

def process_cmath(item):
    item = {
        'dataset': 'cmath',
        'id': item['id'],
        'grade': item['grade'],
        'reasoning_step': item['reasoning_step'],
        'messages': [
            {'role': 'user', 'content': item['question'].strip()},
            {'role': 'assistant', 'content': ''}
        ],
        'answer': item['golden'].strip().replace(",", "")
    }
    yield item

def process_agieval_gaokao_math_cloze(item):
    item = {
        'dataset': 'agieval-gaokao-math-cloze',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': item['question'].strip()},
            {'role': 'assistant', 'content': ''}
        ],
        'answer': [strip_string(ans) for ans in item['answer'].strip().split(";")]
    }
    yield item

def process_agieval_gaokao_mathqa(item):
    question = item['question'].strip()
    options = []
    for option in item['options']:
        option = option.strip()
        assert option[0] == '('
        assert option[2] == ')'
        assert option[1] in 'ABCD'
        option = f"{option[1]}: {option[3:].strip()}"
        options.append(option.strip())
    question = f"{question}\n{options}"
    item = {
        'dataset': 'agieval-gaokao-mathqa',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        "answer": item['label']
    }
    yield item

def process_agieval_gaokao_mathqa_few_shot_cot_test(item):
    question = item['question'].strip().rstrip('\\')
    options = " ".join([opt.strip() for opt in item['options']])
    question = f"{question}\n从以下选项中选择:    {options}"
    item = {
        'dataset': 'agieval-gaokao-mathqa',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        "answer": item['label']
    }
    yield item

def process_minif2f_isabelle(item):
    question = f"(*### Problem\n\n{item['informal_statement'].strip()}\n\n### Solution\n\n{item['informal_proof'].strip()} *)\n\nFormal:\n{item['formal_statement'].strip()}"
    item = {
        'dataset': 'minif2f-isabelle',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        "answer": "placeholder"
    }
    yield item
