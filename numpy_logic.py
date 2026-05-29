import numpy as np
import os

# here we define the arrays for the whole simulation,
# all arrays will be pre-allocated to save time during the simulation

# -------------------------------
# simulation parameters

GRID_SIZE = 20
POPULATION_SIZE = 100
MAX_MINES = 20
MIN_MINES = 10
MAX_MONUMENTS = 20
MIN_MONUMENTS = 10

VISION_RANGE = 2
RADAR_CAPACITY = 10
INSTINCT_RECURRENT_CHANNELS = 10
DECISION_RECURRENT_CHANNELS = 2
MAX_HISTORY_LENGTH = 50


# -------------------------------
def generate_main_arrays():
    # main arrays are:the grid,the population_cord,the mines_cord,the monuments_cord
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    population_cord = np.zeros((POPULATION_SIZE, 2), dtype=int)
    mines_cord = np.zeros((MAX_MINES, 2), dtype=int)
    monuments_cord = np.zeros((MAX_MONUMENTS, 2), dtype=int)
    return grid, population_cord, mines_cord, monuments_cord


def calculate_input_size():
    vision_input_size = (
        2 * VISION_RANGE + 1
    ) ** 2  # the number of cells in the vision range
    radar_input_size = RADAR_CAPACITY * 2  # each mine/monument has x and y coordinates
    internal_state_size = 8  #  2 HP, 4 positions (this turn and last turn), 2 wealth
    recurrent_channels = INSTINCT_RECURRENT_CHANNELS + DECISION_RECURRENT_CHANNELS
    total_input_size = (
        vision_input_size + radar_input_size + internal_state_size + recurrent_channels
    )
    return total_input_size


def calculate_output_size():
    # up, down, left, right, it's softmaxed(tanh)>0.4(only for movement outputs) to be valid
    # or else the agent will stay
    # if more then one action is >0.4, the agent will also stay,
    action_output_size = 4
    recurrent_output_size = INSTINCT_RECURRENT_CHANNELS + DECISION_RECURRENT_CHANNELS
    instinct_main_output_size = 2  # 2 hostory indexes between 0-max_hitory_length
    return action_output_size + recurrent_output_size + instinct_main_output_size


def initialize_population_inputs_matrix():
    # this matrix will hold all the inputs for the population, it's shape is (POPULATION_SIZE, total_input_size)
    total_input_size = calculate_input_size()
    population_inputs = np.zeros((POPULATION_SIZE, total_input_size), dtype=np.float32)
    return population_inputs


def initialize_population_outputs_matrix():
    # this matrix will hold all the outputs for the population, it's shape is (POPULATION_SIZE, total_output_size)
    total_output_size = calculate_output_size()
    population_outputs = np.zeros(
        (POPULATION_SIZE, total_output_size), dtype=np.float32
    )
    return population_outputs


def initialize_population_history_matrix():
    # this matrix will hold the history of the population,
    # it's shape is (POPULATION_SIZE, MAX_HISTORY_LENGTH, input_size+output_size)
    total_input_size = calculate_input_size()
    total_output_size = calculate_output_size()
    population_history = np.zeros(
        (POPULATION_SIZE, MAX_HISTORY_LENGTH, total_input_size + total_output_size),
        dtype=np.float32,
    )
    return population_history


def initialize_learning_inputs_matrix():
    # this matrix will hold the learning inputs for the population,
    # it's shape is (POPULATION_SIZE,(input_size+output_size)*2)
    # 2 for the instinct indexes that will be used for learning
    total_input_size = calculate_input_size()
    total_output_size = calculate_output_size()
    learning_inputs = np.zeros(
        (POPULATION_SIZE, (total_input_size + total_output_size) * 2),
        dtype=np.float32,
    )
    return learning_inputs


def initialize_simulation_key_arrays():
    # here we we will add statistical and main simulation arrays
    HP_array = np.zeros(POPULATION_SIZE, dtype=int)
    wealth_array = np.zeros(POPULATION_SIZE, dtype=int)
    accumulated_wealth_array = np.zeros(POPULATION_SIZE, dtype=int)
    fitness_array = np.zeros(POPULATION_SIZE, dtype=float)
    return HP_array, wealth_array, accumulated_wealth_array, fitness_array


# this will also hold the saving and loading logic for the simulation,
def save_simulation_state(
    save_dir,
    generation,
    grid,
    population_cord,
    mines_cord,
    monuments_cord,
    accumulated_wealth_array,
    fitness_array,
):
    index_path = os.path.join(save_dir, "last_generation.npz")
    np.savez(index_path, generation=generation)
    gen_path = os.path.join(save_dir, f"generation_{generation}")
    os.makedirs(gen_path, exist_ok=True)
    np.savez(
        os.path.join(gen_path, "main_arrays.npz"),
        grid=grid,
        population_cord=population_cord,
        mines_cord=mines_cord,
        monuments_cord=monuments_cord,
    )
    np.savez(
        os.path.join(gen_path, "stat_arrays.npz"),
        accumulated_wealth_array=accumulated_wealth_array,
        fitness_array=fitness_array,
    )


def load_simulation_state(save_dir, generation=-1):
    if generation == -1:
        index_path = os.path.join(save_dir, "last_generation.npz")
        if not os.path.exists(index_path):
            raise FileNotFoundError("No saved simulation state found.")
        data = np.load(index_path)
        generation = data["generation"]
    gen_path = os.path.join(save_dir, f"generation_{generation}")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"No saved state for generation {generation} found.")
    main_arrays = np.load(os.path.join(gen_path, "main_arrays.npz"))
    stat_arrays = np.load(os.path.join(gen_path, "stat_arrays.npz"))
    return (
        generation,
        main_arrays["grid"],
        main_arrays["population_cord"],
        main_arrays["mines_cord"],
        main_arrays["monuments_cord"],
        stat_arrays["accumulated_wealth_array"],
        stat_arrays["fitness_array"],
    )
