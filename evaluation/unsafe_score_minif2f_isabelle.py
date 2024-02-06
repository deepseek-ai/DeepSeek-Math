import argparse
import logging
import json
import sys
import os
import time
from tqdm import tqdm
import traceback

class Checker(object):
    """A modified version of the Draft, Sketch, Prove proof-checking client.
    (https://github.com/albertqjiang/draft_sketch_prove/blob/main/autoformalization/checker.py)

    This checker supports Isabelle2022 via PISA
    (https://albertqjiang.github.io/Portal-to-ISAbelle/).

    It supports checking a miniF2F-style proof via `check`.

    Finally, it replaces `sledgehammer` with a call to `normalhammer`.
    """
    def __init__(self, working_dir, isa_path, theory_file, port=9000):
        sys.path.append(os.environ['PISA_PATH'])
        try:
            from pisa_client import initialise_env
            self.initialise_env = initialise_env
        except Exception as e:
            traceback.print_exc()
            print(e)
            print("Set $PISA_PATH to /yourpath/to/Portal-to-ISAbelle/src/main/python")

        self.working_dir = working_dir
        self.isa_path = isa_path
        self.theory_file = theory_file
        self.port = port

    def _initialize(self):
        env = self.initialise_env(
            self.port,
            isa_path=self.isa_path,
            theory_file_path=self.theory_file,
            working_directory=self.working_dir
        )
        return env

    def _exit(self, env):
        try:
            env.post('exit')
        except:
            print("env.post('exit') timed out")
            pass
        os.system("ps aux | grep Isabelle2022/contrib | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
        os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")

    def _parse_output(self, obs):
        """Parse the sledgehammer output, otherwise return an empty string"""
        if '<hammer>' in obs:
            output = obs.split('<hammer>')[0]
        else:
            output = ''
        return output

    def _run_step(self, step, i, tls_name, env):
        obs, reward, done, metadata = env.step_to_top_level_state(
            action=step,
            tls_name=tls_name,
            new_name='default_%d' % i
        )
        error = None
        if 'error:' in obs or 'Step error' in obs or 'Unknown error' in obs:
            error = obs
        return obs, reward, done, metadata, error

    def _run_sledgehammer(self, step, i, tls_name, env):
        # First try heuristics
        for heuristic in [
            'by auto', 'by simp', 'by blast', 'by fastforce',
            'by force', 'by eval', 'by presburger', 'by sos',
            'by arith', 'by linarith', 'by (auto simp: field_simps)'
        ]:
            step_ = step.replace('normalhammer', heuristic)
            obs, reward, done, metadata, error = self._run_step(step_, i, tls_name, env)
            if error is None:
                obs = '%s <hammer> %s' % (heuristic, obs)
                return obs, reward, done, metadata, error
        # Try sledgehammer
        out = self._run_step(step, i, tls_name, env)
        return out

    def check(self, statement_and_proof):
        # Initialize environment
        env = self._initialize()
        env.initialise()

        # Wrap and parse theorem
        theory = Checker.wrap_theorem(statement_and_proof)
        steps = Checker.get_parsed(env, theory)

        result = self._check(env, steps)
        return result

    def _check(self, env, steps):
        done = False
        reason = ''
        success = False
        step_results = []
        tls_name = 'default'
        for i, step in enumerate(steps):
            try:
                time0 = time.time()
                if 'normalhammer' in step:
                    obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
                else:
                    obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                step_time = time.time() - time0
                step_results.append(dict(
                    index=i, step=step, output=self._parse_output(obs), step_time=step_time
                ))
                if error is not None:
                    reason = error
                    success = False
                    done = False
                    break
            except:
                # Timeout - end the proof attempt
                success = False
                done = False
                reason = 'timeout (%d)' % len(step_results)
                step_results.append(dict(index=i, step=step, output=''))
                break

            # Change when successful
            tls_name = 'default_%d' % i

        if done and reward == 1.0:
            success = True

        result = {
            'success': success,
            'reason': reason,
            'num_steps': len(steps),
            'last_step': len(step_results),
            'step_results': step_results
        }
        # Exit environment
        self._exit(env)
        return result

    @staticmethod
    def wrap_theorem(theorem):
        return 'theory Interactive imports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" \n begin\n%s' % theorem

    @staticmethod
    def get_parsed(env, theory, tls_name='default'):
        # The parsing doesn't work well with `normalhammer`, so we replace
        # all hammer calls with sorry, then replace sorry to normalhammer after parsing.
        theory = theory.replace('sledgehammer', 'sorry')
        theory = theory.replace('normalhammer', 'sorry')

        steps = env.post(f"<parse text> ${theory}")
        steps = steps.split('<SEP>')
        steps = [s for s in steps if s.strip() != '']
        # remove '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        steps = [s.replace('sorry', 'normalhammer') for s in steps]
        return steps


def check_proof(formal_statement, proof, working_dir, isa_path, theory_file, port):
    checker = Checker(
        working_dir=working_dir,
        isa_path=isa_path,
        theory_file=theory_file,
        port=port
    )
    theorem_with_proof = f"{formal_statement}\n{proof}"
    result = checker.check(theorem_with_proof)
    return result


def main(args):
    with open(args.output) as f:
        docs = json.load(f)

    if args.limit:
        limit = args.limit
    else:
        limit = len(docs)

    pass_at_1s = []
    pass_at_anys = []
    for i, doc in enumerate(tqdm(docs[:limit])):
        formal_statement = doc['messages'][-2]['content'].split("Formal:", 1)[1].strip()
        proofs = [doc['prediction'].strip()]

        pass_at_1 = 0
        pass_at_any = 0
        checked_proofs = []
        for j, proof in enumerate(proofs):
            result = check_proof(
                formal_statement=formal_statement,
                proof=proof,
                working_dir=args.working_dir,
                isa_path=args.isa_path,
                theory_file=args.theory_file,
                port=args.port
            )

            if result['success']:
                pass_at_any = 1
                if j == 0:
                    pass_at_1 = 1

            checked_proofs.append({
                'proof': proof,
                'result': result
            })

        pass_at_1s.append(pass_at_1)
        pass_at_anys.append(pass_at_any)

        print(f"acc: {sum(pass_at_1s)} / {len(pass_at_1s)} = {sum(pass_at_1s) / max(len(pass_at_1s), 1)}", flush=True)

        doc['eval'] = {
            'checked_proofs': checked_proofs,
            'pass_at_1': pass_at_1,
            'pass_at_any': pass_at_any
        }

    metrics = {
        "pass_at_1": sum(pass_at_1s) / len(pass_at_1s),
        "pass_at_any": sum(pass_at_anys) / len(pass_at_anys),
        "n_samples": len(pass_at_1s)
    }

    output_path = args.output + ".eval"
    metrics_path = os.path.join(os.path.dirname(args.output), "metrics.json.eval")
    json.dump(docs, open(output_path, "w"), indent=4)
    json.dump(metrics, open(metrics_path, "w"), indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.critical(
            "THIS PROGRAM EXECUTES UNTRUSTED MODEL GENERATED CODE."
            "THERE HAS BEEN NO EFFORT TO AVOID OS AND NETWORK SIDE EFFECTS."
            "USE WITH CAUTION."
    )

    parser = argparse.ArgumentParser("Unsafe script for scoring the minif2f_isabelle tasks")

    parser.add_argument(
        "--isa-path",
        type=str,
        help="path to Isabelle installation (see setup documentation), e.g. "
             "/path/to/Isabelle2022"
    )
    parser.add_argument(
        "--theory-file",
        type=str,
        help="path to Interactive.thy (see setup documentation), e.g. "
             "/path/to/Isabelle2022/src/HOL/Examples/Interactive.thy"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        help="path to Isabelle working directory (see setup documentation), e.g. "
             "/path/to/Isabelle2022/src/HOL/Examples"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="PISA server port (see setup documentation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to output file from running miniF2F Isabelle tasks"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="for debugging purposes, max examples per task to process"
    )

    args = parser.parse_args()
    main(args)