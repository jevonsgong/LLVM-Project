import ast
import re
import rpyc

from typing import Dict, Tuple

from intercode.envs.ic_env import (
    IntercodeEnv,
    ACTION_EXEC, AGENT_OBS, EVAL_OBS, REWARD
)

HOST_PORT = 3006
RESET_KEYWORD = "RESET_CONTAINER_SPECIAL_KEYWORD"

class PythonEnv(IntercodeEnv):
    """Gym environment for python shell"""
    name = "ic_python"

    def __init__(self, image_name: str, **kwargs):
        kwargs['ports'] = {f"{HOST_PORT}/tcp": HOST_PORT}
        super(PythonEnv, self).__init__(image_name, **kwargs)
        self.conn = rpyc.connect("localhost", HOST_PORT)
        self.is_agent = kwargs.get("is_agent", False)
        self.reward_history = []
    
    def reset_container(self) -> None:
        self.conn.root.execute(RESET_KEYWORD)
    
    def exec_action(self, action: str) -> None:
        try:
            if action.strip().startswith("def "):
                if not self.is_agent:
                    function_definition = self.input_multiline_function()
                    action = action + "\n" + function_definition
            else:
                action = self.wrap_with_print(action)
            self.logger.info(f"Command run: {action}")
            self.observation = self.conn.root.execute(action)
            self.info[ACTION_EXEC] = 'error' in self.observation and len(self.observation['error']) > 0
        except Exception as err:
            self.observation = f"Error executing action: {err}"
            self.info[ACTION_EXEC] = False
    
    def get_reward(self) -> Tuple[float, Dict]:
        MAP_DATASET_TO_REWARD = {
            "ic_apps": self.get_reward_apps,
            "ic_mbpp": self.get_reward_mbpp,
            "ic_mbpp_small": self.get_reward_mbpp,
        }
        dataset = self.data_path.split("/")[-1].split(".")[0]

        return MAP_DATASET_TO_REWARD[dataset]()
    
    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.logger.info("Agent container stopped")
    
    ############################
    ### MARK: Helper methods ###
    ############################
    def input_multiline_function(self):
        lines = []
        while True:
            line = input(". ")
            if len(line) == 0:
                break
            lines.append(line)
        return "\n".join(lines)
    
    def wrap_with_print(self, command):
        # Parse the command as an AST (Abstract Syntax Tree)
        parsed_command = ast.parse(command.strip())

        # Check if the command contains an assignment node, print node, or import
        has_assignment = any(isinstance(node, ast.Assign) for node in ast.walk(parsed_command))
        has_print = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print' for node in ast.walk(parsed_command))
        has_import = any(isinstance(node, ast.Import) for node in ast.walk(parsed_command))
        is_assert = command.strip().startswith("assert")

        # Wrap the command with "print" if it's not an assignment and does not have a "print" statement
        if not any([has_assignment, has_print, has_import, is_assert]):
            return f"print({command})"
        else:
            return command
    
    ##############################
    ### MARK: Reward functions ###
    ##############################
    def get_reward_apps(self):
        self.info = {}
        return 0.0, self.info

    def get_reward_mbpp(self):
        self.info = {}

        # Get function from `submit` action
        # TODO: Assert that function name is given upon `submit` action
        last_action = self.trajectory[-1][0]
        #print(self.trajectory)
        func_name = re.search(r'def (\w+)\s*\(', last_action).group(1)

        # Get gold function name, assign to submitted function
        func_name_ref = re.search(r'def (\w+)\s*\(', self.gold).group(1)
        self.conn.root.execute(f"{func_name_ref} = {func_name}")

        # Run tests against submitted function
        results_pred = {}
        self.conn.root.execute(self.record["test_setup_code"])
        for test in self.record["tests"]:
            results_pred[test] = self.conn.root.execute(test)

        # Load gold + run tests
        results_gold = {}
        self.conn.root.execute(RESET_KEYWORD)
        self.conn.root.execute(self.record["test_setup_code"])
        self.conn.root.execute(self.gold)
        for test in self.record["tests"]:
            results_gold[test] = self.conn.root.execute(test)
        
        self.info["submitted_function"] = func_name
        self.info[AGENT_OBS] = results_pred
        self.info[EVAL_OBS] = results_gold

        # Compute reward
        correct = 0
        for test, output in results_pred.items():
            output_gold = results_gold[test]
            if output == output_gold:
                correct += 1
        reward = float(correct) / len(results_pred)
        if len(self.trajectory) > 1:
            action_lst = [self.trajectory[i][0] for i in range(0,len(self.trajectory)-1)]
            if last_action in action_lst and reward < 1:
                reward *= 0.6
            elif self.reward_history and reward <= self.reward_history[-1]/0.9:
                reward *= 0.9
        self.reward_history.append(reward)
        self.info[REWARD] = reward
        self.reward = self.info[REWARD]
        # Reward redesign might also need to change templates, i.e. explain reward calc?
        # Punish if it goes into trap/does not improve
        # if all correct, reward must be 1.0

        self.logger.info(f"Info: {self.info}")
        self.logger.info(f"Reward: {self.reward}")
        return self.reward, self.info