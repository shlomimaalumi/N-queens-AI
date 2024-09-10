# **Algorithmic Solutions Project**

Welcome to the **Algorithmic Solutions Project**, a collection of algorithms aimed at solving optimization problems. The
project includes multiple algorithms such as the Naive Algorithm, Min Conflicts Algorithm, Genetic Algorithm, and an
advanced Mutation Rate Fitting algorithm which features an adaptive mutation rate prediction.

This project is designed to be flexible, allowing you to run different algorithms from the command line, choose the best
mutation rate for genetic algorithms, and visualize the results using plots. This guide will walk you through setting up
the environment, installing dependencies, and running the algorithms.

## **Table of Contents**

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Algorithms](#running-algorithms)
    - [Naive Algorithm](#running-the-naive-algorithm)
    - [Min Conflicts Algorithm](#running-the-min-conflicts-algorithm)
    - [Genetic Algorithm](#running-the-genetic-algorithm)
    - [Mutation Rate Fitting](#running-mutation-rate-fitting)
    - [Adaptive Mutation Rate](#running-adaptive-mutation-rate-prediction)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## **Introduction**

The **Algorithmic Solutions Project** implements several key algorithms commonly used in optimization and
problem-solving. Whether you are trying to solve puzzles, optimize constraints, or fine-tune genetic algorithms, this
project can help you experiment with different approaches.

### Algorithms Included:

- **Naive Algorithm**: A simple, brute-force approach to solving optimization problems.
- **Min Conflicts Algorithm**: A heuristic-based algorithm that seeks to minimize conflicts, commonly used in constraint
  satisfaction problems like the N-Queens puzzle.
- **Genetic Algorithm**: A biologically inspired algorithm used for optimization, incorporating mutation, crossover, and
  selection.
- **Mutation Rate Fitting**: A technique for dynamically finding the best mutation rate in genetic algorithms using a
  predictive approach.

## **Prerequisites**

Before you can run this project, ensure you have the following installed on your system:

- **Python** (version 3.8 or higher)
- **pip** (Python package installer)

You will also need several Python libraries, which are listed in the `requirements.txt` file. We will walk through how
to install them in the next section.

## **Project Structure**

Here’s a breakdown of the key files in this project:

- `main.py`: The main file that provides a command-line interface to run the different algorithms.
- `naive_algorithm.py`: Implements the Naive Algorithm.
- `MIN_CONFLICTS_algorithm.py`: Implements the Min Conflicts Algorithm.
- `Genetic_algorithm.py`: Implements the Genetic Algorithm.
- `mutation_rate_fit.py`: Implements the Mutation Rate Fitting and adaptive mutation rate prediction.
- `plots.py`: Generates visualizations for the results.
- `requirements.txt`: Lists all the dependencies required by the project.

## **Installation**

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/algorithmic-solutions.git
   cd algorithmic-solutions
2. **Install dependencies:**

The required dependencies are listed in the requirements.txt file. Use the following command to install them:
    
```bash
pip install -r requirements.txt
```
    
3. **Verify Installation:**
You can verify the installation by running: 

```bash
python main.py --help
```
This should display the available options for running the algorithms.

## **Running Algorithms**

This section will guide you through how to run each algorithm using the provided main.py script.

**Running the Naive Algorithm**
To run the Naive Algorithm, use the following command:
    
```powershell
python main.py naive --problem nqueens --size 8
```





