import os
import io
from contextlib import redirect_stdout
import pickle
import regex
import copy
from typing import Any, Dict, Optional
import multiprocess
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from functools import partial
import traceback
from timeout_decorator import timeout

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout

    def process_generation_to_code(self, gens: str):
        batch_code = []
        for g in gens:
            multiline_comments = False
            code = []
            for line in g.split('\n'):
                strip_line = line.strip()
                if strip_line.startswith("#"):
                    line = line.split("#", 1)[0] + "# comments"
                elif not multiline_comments and strip_line.startswith('"""') and strip_line.endswith('"""') and len(strip_line) >= 6:
                    line = line.split('"""', 1)[0] + '"""comments"""'
                elif not multiline_comments and strip_line.startswith('"""'):
                    multiline_comments = True
                elif multiline_comments and strip_line.endswith('"""'):
                    multiline_comments = False
                    line = ""
                if not multiline_comments:
                    code.append(line)
            batch_code.append(code)
        return batch_code

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        try:
            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                program_io.seek(0)
                result = "".join(program_io.readlines()) # [-1]
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            concise_exec_info = ""
            exec_info = ""
            str(result)
            pickle.dumps(result) # serialization check
        except:
            # traceback.print_exc()
            result = ''
            concise_exec_info = traceback.format_exc().split('\n')[-2]
            exec_info = traceback.format_exc()
            if get_answer_from_stdout and 'exec(code_piece, self._global_vars)' in exec_info:
                exec_info = exec_info.split('exec(code_piece, self._global_vars)')[-1].strip()
                msg = []
                for line in exec_info.split("\n"):
                    patt = regex.search(r'(?P<start>.*)File "(?P<file>.*)", line (?P<lno>\d+), (?P<end>.*)', line)
                    if patt is not None:
                        if '<module>' in patt.group('end'):
                            continue
                        fname = patt.group("file")
                        if "site-packages" in fname:
                            fname = f"site-packages{fname.split('site-packages', 1)[1]}"
                            line = f'{patt.group("start")}File "{fname}", {patt.group("end")}'
                        else:
                            line = f'{patt.group("start")}{patt.group("end")}'
                    else:
                        patt = regex.search(r'(?P<start>.*)(?P<file>/.*site-packages/.*\.py)(?P<end>.*)', line)
                        if patt is not None:
                            line = f'{patt.group("start")}site-packages{patt.group("file").split("site-packages", 1)[1]}{patt.group("end")}'
                    msg.append(line)
                exec_info = "\n".join(msg)
        return result, concise_exec_info, exec_info

    def apply(self, code):
        return self.batch_apply([code])[0]

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)
        all_exec_results = []
        executor = partial(
            self.execute,
            get_answer_from_stdout=self.get_answer_from_stdout,
            runtime=self.runtime,
            answer_symbol=self.answer_symbol,
            answer_expr=self.answer_expr,
            timeout_length=10,
        )
        with ProcessPool(max_workers=multiprocess.cpu_count()) as pool:
            iterator = pool.map(executor, all_code_snippets, timeout=10).result()

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    all_exec_results.append(("", "Timeout Error", "Timeout Error"))
                except Exception as error:
                    print(error)
                    exit()

        batch_results = []
        for code, (result, concise_exec_info, exec_info) in zip(all_code_snippets, all_exec_results):
            metadata = {'code': code, 'exec_result': result, 'concise_exec_info': concise_exec_info, 'exec_info': exec_info}
            batch_results.append((result, metadata))
        return batch_results
