import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/Users/gongwenzhen/anaconda3/envs/intercode/lib/python3.9/site-packages")
import numpy as np
import argparse, json, os, re
from intercode.envs import (
    BashEnv, CTFEnv, PythonEnv, SqlEnv, ACTION_EXEC, AGENT_OBS
)
from tqdm import tqdm
from typing import Dict, List
from experiments.policies import (
    CompletionGPTPolicy, ChatGPTPolicy, PalmChatPolicy, PalmCompletionPolicy
)
from experiments.utils import HANDICAP_MAP, PROMPT_MAP
from rich import print
import networkx as nx
from sklearn.cluster import SpectralClustering
import textwrap
from scipy.sparse.csgraph import laplacian
from experiments.utils.gpt_api import ChatGPT,CompletionGPT
from experiments.utils import ACTION_PARSER_MAP

parser = argparse.ArgumentParser(description='N-turn evaluation for Intercode environment')
parser.add_argument('--data_path', type=str, help='path to dataset to evaluate on')
parser.add_argument('--env', choices=['sql', 'bash', 'python', 'ctf'], help='Intercode environment to run eval on')
parser.add_argument('--image_name', type=str, help='name of docker image to build environment with')
parser.add_argument('--log_dir', type=str, help='folder to save experiment run log file to')
parser.add_argument('--max_turns', type=int, help='max number of interaction turns')
parser.add_argument('--verbose', action='store_true', help="print out logs")
parser.add_argument('--model', type=str, help="model to use for policy")
args = parser.parse_args()

SETTING_MAP = {
    "sql": "MySQL Database",
    "bash": "Bourne Shell",
    "python": "Python 3 Interpreter",
    "ctf": "Capture the Flag"
}





def isfloat(str):
    try:
        float(str)
    except ValueError:
        return False
    return True




class GPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        self.messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
        ]
        self.action_parser = ACTION_PARSER_MAP[args.env]

    def calc_similarity(self, a, b):
        messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
            {"role": "user",
             "content": f"On a scale from 0 to 10, what is the logical relationship between the following two sentences? 0 represents being totally contradicting, 10 represents being totally the same, and 5 represents they are independent. your answer should only return a real number without any other text. your answer should only return a real number without any other text.\nsentence 1:{a};\nsentence 2:{b}"}
        ]
        response = ChatGPT(messages=messages)
        result = response[0]
        messages.append({"role": "assistant", "content": result})
        count = 0
        while not isfloat(result):
            count += 1
            if count > 5:
                return 0
                # raise ValueError
            self.messages.append({"role": "user", "content": "give a numerical answer only"})
            response = ChatGPT(messages=messages)
            result = response[0]

        return result

    def aggregate_thoughts(self, problem, thoughts):
        messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
            {"role": "user",
             "content": f"You are trying to solve this problem: {problem}\nHere is a subset of belief you generated:{';'.join(thoughts)}. Now you aggregate these thoughts into a single thoughts, note that do not attempt to solve this problem yet, just aggregate the thoughts, make sure you include the details. Return your answer directly."}
        ]
        response = ChatGPT(messages=messages)
        return response[0]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = ChatGPT(messages=self.messages)
        self.messages.append({"role": "assistant", "content": response[0]})
        return response[0]

    def chat_onetime(self, message):
        messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
        ]
        messages.append({"role": "user", "content": message})
        response = ChatGPT(messages=self.messages)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response[0]

    def reset(self):
        self.messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure. "},
        ]


class solver:
    def __init__(self):
        self.bot1 = GPT()
        self.bot2 = GPT()
        self.setting = SETTING_MAP[args.env]
        self.questions_understanding = [
            "How many and which specific elements or objects are involved in this problem?",
            "What are the intrinsic properties and characteristics of these elements or objects?",
            "What is the relationship or interaction between these elements or objects?"]

    def cot_prompt(self, problem, thoughts):
        return f"The question {problem}, Some fact you might find useful include: {thoughts}. Now lets plan to solve this problem step by step. After that, revise your solution, and give me the final result."

    def similarity_score(self, thoughts1, thoughts2):
        response = self.bot2.calc_similarity(thoughts1, thoughts2)
        score = np.float32(response)
        if score < 0:
            score = np.float32(0)
        return score

    def generate_matrix(self, thoughts):
        n = len(thoughts)
        similarity_matrix = np.zeros((n, n))

        # Compute similarity scores
        for i in range(n):
            for j in range(i + 1, n):  # Skip diagonal and redundant computations
                score = self.similarity_score(thoughts[i], thoughts[j])
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score

        np.fill_diagonal(similarity_matrix, 0)

        return similarity_matrix

    def aggregate(self, problem, thoughts, labels):
        unique_labels = np.unique(labels)

        aggregated_thoughts = []

        for label in unique_labels:
            # Find the subset of thoughts corresponding to the current label
            subset_thoughts = [thought for thought, lbl in zip(thoughts, labels) if lbl == label]

            # Call the aggregate_thoughts method and store the result
            aggregated_thought = self.bot2.aggregate_thoughts(problem, subset_thoughts)
            aggregated_thoughts.append(aggregated_thought)

        # Print the number of unique labels
        # print(f"Number of labels: {len(unique_labels)}")

        return aggregated_thoughts

    def spectral_clustering_auto_select(self, matrix):
        # Compute the Laplacian
        L, _ = laplacian(matrix, normed=True, return_diag=True)
        eigenvalues = np.linalg.eigvalsh(L)

        eigenvalues.sort()

        gaps = np.diff(eigenvalues)
        k = np.argmax(gaps) + 1
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
        labels = clustering.fit_predict(matrix)

        return labels, k

    def thought_clustering(self, thoughts):
        mat = self.generate_matrix(thoughts)
        n_clusters = 3
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = clustering.fit_predict(mat)
        # labels, n_clusters=self.spectral_clustering_auto_select(mat)

        new_mat = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    indices_i = np.where(labels == i)[0]
                    indices_j = np.where(labels == j)[0]
                    sum_weights = np.sum(mat[np.ix_(indices_i, indices_j)])
                    n_edges = len(indices_i) * len(indices_j)
                    new_mat[i, j] = sum_weights / n_edges if n_edges > 0 else 0

        return mat, labels, new_mat

    def understand(self, problem, num_thoughts):
        filtered_thoughts = []

        for question in self.questions_understanding:
            # Ask each question individually
            response = self.bot1.chat(
                f"The problem statement you are trying to solve is: '{problem}'. Please answer the following question: {question}return the answers only. Return exacctly {num_thoughts // 3} bullet points for this question. your answer should be in full sentence. Your answer should be separated by '\n'")
            # Extract the answer
            thoughts = response.split("\n")
            # Filter out short responses
            valid_thoughts = [element for element in thoughts if len(element) >= 2]

            # Add valid thoughts to the final list
            filtered_thoughts.extend(valid_thoughts)

            # Break the loop if the number of desired thoughts is reached
            if len(filtered_thoughts) >= num_thoughts:
                break

        # Return the first 'num_thoughts' thoughts
        return filtered_thoughts[:num_thoughts]

    def get_retry_msg(self):
        return f"""No executable {args.env} code was found in your last response."""

    def get_obs_msg(self, observation, reward):
        if isinstance(observation, str) and observation == "" or isinstance(observation, list) and len(
                    observation) == 0:
            observation = "No output"
        return f"""{self.setting} Output: {observation}
    Reward: {reward}
    Here is the query again: \"{self.query}\"
    Try something different to generate {args.env} command to get a reward of 1.
    Do not generate any output or reward.
    """

    def run(self, problem):
        try:
            self.query = problem
            self.bot1.reset()
            self.bot2.reset()
            thoughts = self.understand(problem, 15)
            # print(thoughts)
            mat, label, mat_agg = self.thought_clustering(thoughts)
            # self.visualize_graph("before_agg", thoughts, mat, label)
            aggregated_thoughts = self.aggregate(problem, thoughts, label)
            # print()
            # print(aggregated_thoughts)
            # self.visualize_graph("after_agg", aggregated_thoughts, mat_agg, np.unique(label))
            self.bot2.reset()
            self.bot2.chat(self.cot_prompt(problem, aggregated_thoughts))
            # print(self.bot2.messages)
            return self.bot2.chat("now return the executable code part of the answer only, do not return anything other then numbers")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return 0

class ExperimentWrapper():
    def __init__(self, args):
        self.args = args
        self.action_parser = ACTION_PARSER_MAP[args.env]
        # Set environment (No logging for env)
        self.env = None
        if args.env == 'bash':
            self.env = BashEnv(image_name=args.image_name,
                               data_path=args.data_path)
        elif args.env == 'python':
            self.env = PythonEnv(image_name=args.image_name,
                                 data_path=args.data_path, is_agent=True)
        else:
            raise ValueError(f'Environment {args.env} not recognized')

        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        log_file_name = f"{self.env.name}_solver.json"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        self.log_data = {}

        # Initialize Policy
        self.policy = solver()

    def run_expr(self):
        try:
            for idx in tqdm(range(0, len(self.env.data_loader)), disable=self.args.verbose):
                # Reset variables per task
                self.env.reset(idx)
                observation, reward, valid_action = None, None, None
                turn_history = {"actions": [], "observations": [], "rewards": [], "valid_action": []}
                record = self.env.data_loader.get(idx)

                if self.args.verbose:
                    print(f'------\nQuery {idx}: {self.env.query}')
                for turn in range(self.args.max_turns):
                    # Invoke Action -> Observation Iteration
                    try:
                        action = self.policy.run(self.env.query)
                        action, is_code = self.action_parser(action)
                    except (ValueError, TypeError) as e:
                        print(f"[ERROR] Index {idx}: {e}")
                        # Logging
                        turn_history["actions"].append("blocked")
                        turn_history["rewards"].append(0)
                        break

                    if not is_code:
                        reward = 0
                        observation = self.policy.get_retry_msg()
                        valid_action = False
                    else:
                        if isinstance(self.env, PythonEnv):
                            if action.startswith("def"):
                                func_name = re.match(r'def (\w+)\(', action).group(1)
                                observation, reward, _, info = self.env.step(action)
                                _, reward, _, info = self.env.step("submit " + func_name)

                                SHOW_FAILED_CASE = 0
                                if reward != 1:
                                    if SHOW_FAILED_CASE == 1:
                                        for k, v in info[AGENT_OBS].items():
                                            if len(v['error']) > 0:
                                                observation = f"Failed Test Case: {k}\nPlease try again."
                                                break
                                    elif SHOW_FAILED_CASE == 2:
                                        fails = 0
                                        for k, v in info[AGENT_OBS].items():
                                            if len(v['error']) > 0:
                                                fails += 1
                                        observation = f"Failed {fails}/{len(info[AGENT_OBS])} Test Cases. Please try again."
                                    else:
                                        if any([len(v['error']) > 0 for k, v in info[AGENT_OBS].items()]):
                                            observation = "Test case did not pass. Please try again."
                            else:
                                observation, reward, _, info = self.env.step(action)
                        else:
                            if action != "skip":
                                observation, reward, done, info = self.env.step(action)
                                valid_action = info[ACTION_EXEC]
                            else:
                                observation, reward, done, info = "skipped", 0, True, {}
                                valid_action = True
                            if not isinstance(self.env, CTFEnv):
                                _, reward, done, info = self.env.step("submit")
                            else:
                                if done and reward != 1 and action.lower() != "skip":
                                    observation = "Submitted flag is incorrect. Keep trying!"

                    if self.args.verbose:
                        print(f"- Turn {turn}")
                        print(f"-- Action: {action}")
                        if isinstance(observation, str) and observation.startswith(f'No {args.env} code'):
                            print(
                                f"-- Observation: (meta) No code output, policy's template's retry message was invoked")
                        else:
                            print(f"-- Observation: {observation}")

                    # Logging
                    turn_history["actions"].append(action)
                    turn_history["observations"].append(str(observation))  # To avoid serialization issues
                    turn_history["rewards"].append(reward)
                    turn_history["valid_action"].append(valid_action)

                    # End episode upon perfect reward
                    if reward == 1 or action.lower() == "skip":
                        break

                max_reward = max(turn_history["rewards"])
                log_episode = {
                    "environment": self.env.name,
                    "dataset": self.args.data_path,
                    "task_id": idx,
                    "query": self.env.query,
                    "turn_history": turn_history,
                    "summary": {
                        "max_reward": max_reward,
                        "max_reward_idx": turn_history["rewards"].index(max_reward),
                        "turns_taken": turn + 1,
                        "turns_max": self.args.max_turns,
                    }
                }
                if "hardness" in record:
                    log_episode["hardness"] = record["hardness"]
                self.log_data[idx] = log_episode

                if self.args.verbose:
                    print(f"Query {idx} Finished\n-Reward: {max_reward}\n-Turns: {turn + 1}")

        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
        finally:
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
            self.env.close()


if __name__ == '__main__':
    expr_wrapper = ExperimentWrapper(args)
    expr_wrapper.run_expr()
