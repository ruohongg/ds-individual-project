import random
import string
import numpy as np
from coverage import Coverage
from test_function import example_function
import json
import matplotlib.pyplot as plt


def generate_random_test_case():
    x = random.randint(-100, 100)
    y = random.uniform(-100.0, 100.0)
    z = ''.join(random.choices(string.ascii_letters, k=random.randint(1, 10)))
    w = [random.randint(-100, 100) for _ in range(random.randint(1, 10))]
    return (x, y, z, w)

def calculate_coverage(test_cases):
    cov = Coverage(include=["test_function.py"])
    cov.start()

    for x, y, z, w in test_cases:
        example_function(x, y, z, w)

    cov.stop()
    cov.save()

    json_report_path = "coverage_report.json"
    cov.json_report(outfile=json_report_path)
    with open(json_report_path, "r") as f:
        json_data = json.load(f)

    return json_data.get("totals", {}).get("percent_covered", 0)

def calculate_fitness(test_case, current_test_cases):
    temp_test_cases = current_test_cases + [test_case]
    return calculate_coverage(temp_test_cases)

def selection(population, fitness_scores, num_to_select):
    selected_indices = np.argsort(fitness_scores)[-num_to_select:]
    return [population[i] for i in selected_indices]

# Crossover
def crossover(parent1, parent2):
    x1, y1, z1, w1 = parent1
    x2, y2, z2, w2 = parent2

    child_x = random.choice([x1, x2])
    child_y = random.choice([y1, y2])
    child_z = random.choice([z1, z2])
    child_w = random.choice([w1, w2])

    return (child_x, child_y, child_z.strip(), child_w)

# Mutation
def mutate(test_case, mutation_rate=0.1):
    x, y, z, w = test_case
    if random.random() < mutation_rate:
        x += random.randint(-10, 10)
    if random.random() < mutation_rate:
        y += random.uniform(-10.0, 10.0)
    if random.random() < mutation_rate:
        z = ''.join(random.choices(string.ascii_letters, k=len(z)))
    if random.random() < mutation_rate:
        w = [random.randint(-100, 100) for _ in range(len(w))]
    return (x, y, z, w)

# GA Implementation
def genetic_algorithm(num_generations=50, population_size=20, mutation_rate=0.1, elite_size=5):
    # Initialization
    population = [generate_random_test_case() for _ in range(population_size)]
    best_test_cases = []
    coverage_history = []

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(tc, best_test_cases) for tc in population]

        best_indices = np.argsort(fitness_scores)[-elite_size:]
        best_test_cases.extend([population[i] for i in best_indices])

        best_coverage = max(fitness_scores)
        coverage_history.append(best_coverage)
        print(f"Generation {generation + 1}: Best Coverage = {best_coverage}%")

        selected_parents = selection(population, fitness_scores, num_to_select=elite_size)

        next_generation = selected_parents[:]
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation

    return best_test_cases, coverage_history

def plot_coverage(coverage_history):
    plt.figure(figsize=(10, 6))
    plt.plot(coverage_history, label='Coverage Progress')
    plt.title('Coverage Progression Across Generations', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Coverage (%)', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

best_test_cases, coverage_history = genetic_algorithm()

with open("best_test_cases.json", "w") as f:
    json.dump(best_test_cases, f, indent=4)
print("Best test cases saved to 'best_test_cases.json'.")

plot_coverage(coverage_history)
