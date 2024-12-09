import inspect
import random
import string
import numpy as np
from gymnasium import Env, spaces
from coverage import Coverage
from stable_baselines3 import PPO
from test_function import example_function
import matplotlib.pyplot as plt
import json


# def example_function(x: int, y: float, z: str, w: list):
#     if x > 50:
#         result = y * x
#         if "a" in z:
#             result += len(z)
#         elif "b" in z:
#             result -= len(z)
#         else:
#             result *= len(z)
#     elif x < 0:
#         result = sum(w) + y
#         if any(i > 10 for i in w):
#             result += 10
#         else:
#             result -= 5
#     else:
#         result = y / (x + 1)
#         if z.isdigit():
#             result *= int(z)
#         elif len(z) > 5:
#             result += len(w)
#         else:
#             result -= sum(w)
    
#     if len(w) >= 3 and w[0] == w[-1]:
#         result += sum(w[:3])
#     elif len(w) < 3:
#         result -= len(w) * 2
    
#     if z.startswith("test"):
#         result *= 1.5
#     elif z.endswith("end"):
#         result /= 2
    
#     return result

def analyze_function(func):
    signature = inspect.signature(func)
    params = {}
    for name, param in signature.parameters.items():
        annotation = param.annotation
        default = param.default if param.default != inspect.Parameter.empty else None
        params[name] = {"type": annotation, "default": default}
    return params

class TestCaseGeneratorEnv(Env):
    # Group Size -> Predefined
    coverages = []

    def __init__(self, func, group_size = 10, GA = False):
        super(TestCaseGeneratorEnv, self).__init__()
        self.func = example_function
        self.group_size = group_size
        self.previous_coverage = 0
        self.test_cases = []
        self.all_test_cases = []
        self.group_reward = []
        self.threshold = 80
        self.optimized_initial_state = None
        self.GA = GA

        # Parameter's range, Predefined.
        self.param_ranges = {
            "x": [-1, 1],
            "y": [-1.0, 1.0],
            "z": [-1, 1], #lengh of String
            "w": [-1, 1]  #length of list
        }
        
        # Action spaces, the action I can take
        self.action_space = spaces.Box(
            low=np.array([self.param_ranges["x"][0], self.param_ranges["y"][0], self.param_ranges["z"][0], self.param_ranges["w"][0]]),
            high=np.array([self.param_ranges["x"][1], self.param_ranges["y"][1], self.param_ranges["z"][1], self.param_ranges["w"][1]]),
            dtype=np.float32
        )

        # state space
        self.observation_space = spaces.Box(
            low=np.array([self.param_ranges["x"][0], self.param_ranges["y"][0], self.param_ranges["z"][0], self.param_ranges["w"][0], 0, 0, 1]),
            high=np.array([self.param_ranges["x"][1], self.param_ranges["y"][1], self.param_ranges["z"][1], self.param_ranges["w"][1], 1, 1, 100]),
            dtype=np.float32
        )

    
    def reset(self,seed=None):
        self.test_cases = []
        self.group_reward = []
        if self.optimized_initial_state is not None:
            return self.optimized_initial_state, {}
        self.state = np.array([0, 0, 0, 0, 0, 0, self.group_size])
        return self.state,{}
    
    def step(self, action):

        # Noise Added
        x = int(action[0]*100)
        x_noise = int(np.random.normal(0, 5))
        x += x_noise
        x = np.clip(x, -100, 100)
        y = action[1] * 100.0
        y_noise = np.random.normal(0, 5) 
        y += y_noise
        y = np.clip(y, -100.0, 100.0)
        z = ''.join(random.choices(string.ascii_letters, k=int(action[2] + 1) * 5))
        w = [random.randint(-100, 100) for _ in range(int(action[3] + 1) * 5)]
        self.test_cases.append((x, y, z, w))
        group_coverage = self.calculate_coverage(self.test_cases)
        reward = group_coverage - self.previous_coverage
        self.group_reward.append(reward)
        avg_reward = np.mean(self.group_reward)

        # print("group_coverage:{}, prev_corverage:{}, reward:{}".format(group_coverage, self.previous_coverage, reward))
        if len(self.test_cases) >= self.group_size:
            if avg_reward < 0.01 and self.previous_coverage < self.threshold :
                self.group_size = min(self.group_size + 1, 100)
            elif avg_reward > 1 and self.previous_coverage > self.threshold:
                self.group_size = max(self.group_size - 1, 5)
            # print("avg_reward:{}, previous_coverage:{}, group size:{}".format(avg_reward, self.previous_coverage, self.group_size))
            # apply GA
            if self.GA:
                self.test_cases = self.apply_ga(self.test_cases, group_coverage)
                group_coverage = self.calculate_coverage(self.test_cases)
                self.optimized_initial_state = np.array([x, y, len(z), len(w), group_coverage, self.previous_coverage, self.group_size])
                # print("coverage after ga:", group_coverage)
            self.all_test_cases.append(self.test_cases.copy())
            self.previous_coverage = group_coverage
            self.coverages.append(group_coverage)
            done = True
        else:
            done = False
        
        num_cases = len(self.test_cases)
        self.state = np.array([x, y, len(z), len(w), group_coverage, self.previous_coverage, self.group_size])
        return self.state, reward, done, {}, {}

    def calculate_coverage(self, test_cases):
        # Calculate the coverage of the test cases
        cov = Coverage(include=["test_function.py"])
        cov.start()

        for x, y, z, w in test_cases:
            self.func(x, y, z, w)

        cov.stop()
        cov.save()

        json_report_path = "coverage_report.json" 
        cov.json_report(outfile=json_report_path)

        # extract json report
        with open(json_report_path, "r") as f:
            json_data = json.load(f)

        percent_covered = json_data.get("totals", {}).get("percent_covered", 0)
        return percent_covered

    # def apply_ga(self, test_cases, current_coverage, ga_epochs = 5):
    def apply_ga(self, test_cases, current_coverage, ga_epochs=2):
        population = test_cases
        for epoch in range(ga_epochs):
            fitness = [self.calculate_fitness(tc, current_coverage) for tc in population]
            selected = self.selection(population, fitness)
            offspring = []
            while len(offspring) < len(population):
                parent1, parent2 = random.sample(selected, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                if self.calculate_fitness(child, current_coverage) > current_coverage:
                    offspring.append(child)
            population = offspring
        return population

    
    def calculate_fitness(self, test_case, current_coverage):
        temp_test_cases = self.test_cases + [test_case]
        test_coverage = self.calculate_coverage(temp_test_cases)
        return test_coverage
    
    def selection(self, population, fitness):
        sorted_indices = np.argsort(fitness)[-len(population)//2:]
        return [population[i] for i in sorted_indices]
    
    def crossover(self, parent1, parent2):
        x1, y1, z1, w1 = parent1
        x2, y2, z2, w2 = parent2


        child_x = random.choice([x1, x2])
        child_y = random.choice([y1, y2])
        child_z = random.choice([z1, z2])
        child_w = random.choice([w1, w2])

        return (child_x, child_y, child_z, child_w)

    def mutate(self, test_case, mutation_rate=0.1):
        x, y, z, w = test_case
        if np.random.rand() < mutation_rate:
            x += np.random.randint(-10, 10)
            y += np.random.uniform(-10.0, 10.0)
            z = ''.join(random.choices(string.ascii_letters, k=len(z))) 
            w = [np.random.randint(-100, 100) for _ in range(len(w))]
        return (x, y, z, w)

    def save_test_cases(self, file_path="test_cases.json"):
        native_test_cases = convert_to_native(self.all_test_cases)
        with open(file_path, "w") as f:
            json.dump(native_test_cases, f, indent=4)
        print(f"Test cases saved to {file_path}")

# For formatting
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    else:
        return obj

def print_graph(coverages):
    coverages = [(sum(coverages[i:i+10])) / 10 for i in range(0, len(coverages), 10)]
    coverages = coverages[:-2]
    plt.figure(figsize=(10, 6))
    plt.plot(coverages, linestyle='-', label='Coverage Progress')
    plt.title('Coverage Convergence', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Coverage', fontsize=12)
    plt.grid(False)
    plt.legend()
    plt.show()


###############################

env = TestCaseGeneratorEnv(example_function)


model = PPO("MlpPolicy", env, verbose=1, device = "cuda")

# train
num_epochs = 1
num_steps_per_epoch = 10000

model.learn(total_timesteps=num_steps_per_epoch)

print(env.coverages)
print_graph(env.coverages)

obs, _ = env.reset()
idx = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    if done:  # Finished one episode

        # final_coverage = obs[5]  # obs[5] 表示最终 coverage
        print(f"Episode finished. Final Coverage: {obs[5]}")
        # env.save_test_cases(f"generated_test_cases_{idx}.json")
        idx += 1

        obs, info = env.reset()
    # else:
    #     print(f"Action: {action}, Reward: {reward}")