import numpy as np
import random
import os
import time
import pickle
import matplotlib.pyplot as plt
import pygame
import sys
from numba import njit

# --- CONSTANTS & CONFIGURATION ---
GRID_SIZE = 20
NUM_AGENTS = 100
MAX_NUM_MINES = 20
MIN_NUM_MINES = 10
NUM_MINES = MAX_NUM_MINES
DIFFICULTY_CURVE = 1.0005
MAX_NUM_MONUMENTS = 20
MIN_NUM_MONUMENTS = 10
NUM_MONUMENTS = MAX_NUM_MONUMENTS
MAX_TICKS_PER_GEN = 1000
NUM_GENERATIONS = 10000
MUTATION_RATE = 0.1
CHECKPOINT_FILE = "brain_checkpoint02.pkl"
SHUFFLE_INTERVAL = 50

INPUT_SIZE = 66
SHOW_ON = False
CELL_SIZE = 32
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 10


# --- JIT COMPILED CORE MATH ---
@njit
def fast_forward(
    X_input,
    W_conv_inst,
    b_conv_inst,
    W_inst,
    b_inst,
    W_inst_out,
    b_inst_out,
    W_conv_dec,
    b_conv_dec,
    W_dec,
    b_dec,
    W_dec_mov,
    b_dec_mov,
    W_dec_mem,
    b_dec_mem,
    W_mod,
    b_mod,
    W_mod_out,
    b_mod_out,
):
    # 1. Instinct
    x_inst_reshaped = X_input.reshape(-1, 2)
    conv_inst = np.tanh(np.dot(x_inst_reshaped, W_conv_inst) + b_conv_inst)
    conv_inst_flat = conv_inst.flatten()

    h_inst = np.tanh(np.dot(conv_inst_flat, W_inst) + b_inst)
    instinct_vector = np.tanh(np.dot(h_inst, W_inst_out) + b_inst_out)

    # 2. Decision
    x_dec = np.concatenate((X_input, instinct_vector))
    x_dec_reshaped = x_dec.reshape(-1, 2)
    conv_dec = np.tanh(np.dot(x_dec_reshaped, W_conv_dec) + b_conv_dec)
    conv_dec_flat = conv_dec.flatten()

    h_dec = np.tanh(np.dot(conv_dec_flat, W_dec) + b_dec)
    mov_logits = np.dot(h_dec, W_dec_mov) + b_dec_mov
    mem_logits = np.dot(h_dec, W_dec_mem) + b_dec_mem

    # Softmax
    ev_mov = np.exp(mov_logits - np.max(mov_logits))
    mov_probs = ev_mov / ev_mov.sum()

    ev_mem = np.exp(mem_logits - np.max(mem_logits))
    mem_probs = ev_mem / ev_mem.sum()

    chosen_mov = np.argmax(mov_probs)
    chosen_mem = np.argmax(mem_probs)

    # 3. Modulation
    decision_vector = np.concatenate((mov_probs, mem_probs))
    x_mod = np.concatenate((X_input, instinct_vector, decision_vector))

    h_mod = np.tanh(np.dot(x_mod, W_mod) + b_mod)
    mod_out = np.dot(h_mod, W_mod_out) + b_mod_out

    eta = 1.0 / (1.0 + np.exp(-mod_out[0])) * 0.01
    A = np.tanh(mod_out[1])
    B = np.tanh(mod_out[2])

    h_inst_row = h_inst.reshape(1, -1)
    h_dec_row = h_dec.reshape(1, -1)
    h_mod_row = h_mod.reshape(1, -1)

    dW_inst = eta * (A * np.outer(conv_inst_flat, h_inst) + B * h_inst_row)
    dW_dec = eta * (A * np.outer(conv_dec_flat, h_dec) + B * h_dec_row)
    dW_mod = eta * (A * np.outer(x_mod, h_mod) + B * h_mod_row)

    return chosen_mov, chosen_mem, dW_inst, dW_dec, dW_mod


class Agent:
    def __init__(self, agent_id, base_weights=None):
        self.id = agent_id
        self.x = 0
        self.y = 0
        self.health = 200
        self.wealth = 0

        self.prev_x = 0
        self.prev_y = 0
        self.prev_health = 200
        self.prev_wealth = 0

        self.memory = {}
        self.alive = True
        self.ticks_survived = 0
        self.fitness = 0.0
        self.prev_movement = 0

        if base_weights is None:
            self.init_base_weights()
        else:
            self.base_weights = base_weights

        self.reset_lifetime_state()

    def init_base_weights(self):
        instinct_first_layer_size = 48
        decision_first_layer_size = 48
        modulation_first_layer_size = 20

        self.base_weights = {
            "W_conv_inst": np.random.randn(2, 6) * np.sqrt(2.0 / 2),
            "b_conv_inst": np.zeros(6),
            "W_inst": np.random.randn(198, instinct_first_layer_size)
            * np.sqrt(2.0 / 198),
            "b_inst": np.zeros(instinct_first_layer_size),
            "W_inst_out": np.random.randn(instinct_first_layer_size, 4)
            * np.sqrt(2.0 / instinct_first_layer_size),
            "b_inst_out": np.zeros(4),
            "W_conv_dec": np.random.randn(2, 6) * np.sqrt(2.0 / 2),
            "b_conv_dec": np.zeros(6),
            "W_dec": np.random.randn(210, decision_first_layer_size)
            * np.sqrt(2.0 / 210),
            "b_dec": np.zeros(decision_first_layer_size),
            "W_dec_mov": np.random.randn(decision_first_layer_size, 5)
            * np.sqrt(2.0 / decision_first_layer_size),
            "b_dec_mov": np.zeros(5),
            "W_dec_mem": np.random.randn(decision_first_layer_size, 3)
            * np.sqrt(2.0 / decision_first_layer_size),
            "b_dec_mem": np.zeros(3),
            "W_mod": np.random.randn(INPUT_SIZE + 4 + 8, modulation_first_layer_size)
            * np.sqrt(2.0 / (INPUT_SIZE + 4 + 8)),
            "b_mod": np.zeros(modulation_first_layer_size),
            "W_mod_out": np.random.randn(modulation_first_layer_size, 3)
            * np.sqrt(2.0 / modulation_first_layer_size),
            "b_mod_out": np.zeros(3),
        }

    def reset_lifetime_state(self):
        self.health = 200
        self.wealth = 0
        self.prev_health = 200
        self.prev_wealth = 0
        self.prev_x = self.x
        self.prev_y = self.y
        self.memory = {}
        self.alive = True
        self.ticks_survived = 0
        self.fitness = 0.0
        self.prev_movement = 0
        self.weights = {k: v.copy() for k, v in self.base_weights.items()}

    def forward_and_adapt(self, sensory_vector):
        normalized_prev_mov = self.prev_movement / 4.0
        X_input = np.concatenate([sensory_vector, [normalized_prev_mov]])

        chosen_mov, chosen_mem, dW_inst, dW_dec, dW_mod = fast_forward(
            X_input,
            self.weights["W_conv_inst"],
            self.weights["b_conv_inst"],
            self.weights["W_inst"],
            self.weights["b_inst"],
            self.weights["W_inst_out"],
            self.weights["b_inst_out"],
            self.weights["W_conv_dec"],
            self.weights["b_conv_dec"],
            self.weights["W_dec"],
            self.weights["b_dec"],
            self.weights["W_dec_mov"],
            self.weights["b_dec_mov"],
            self.weights["W_dec_mem"],
            self.weights["b_dec_mem"],
            self.weights["W_mod"],
            self.weights["b_mod"],
            self.weights["W_mod_out"],
            self.weights["b_mod_out"],
        )

        self.prev_movement = chosen_mov

        self.weights["W_inst"] += dW_inst
        self.weights["W_dec"] += dW_dec
        self.weights["W_mod"] += dW_mod

        return chosen_mov, chosen_mem


class EvolutionaryEcosystem:
    def __init__(self):
        self.mines = []
        self.monuments = []
        self.mine_set = set()
        self.monument_set = set()
        self.all_opps = []

        self.oasis_tiles = set()
        self.closest_10_map = {}
        self.mine_proximity_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

        self.opportunities_array = np.array([])
        self.grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.padded_grid = np.zeros((GRID_SIZE + 4, GRID_SIZE + 4), dtype=np.float32)

        self.agents = self.load_or_initialize_population()
        self.generate_world_geometry()
        self.reposition_agents()

    def load_or_initialize_population(self):
        if os.path.exists(CHECKPOINT_FILE):
            print(f"Loading genetic lineage from {CHECKPOINT_FILE}...")
            with open(CHECKPOINT_FILE, "rb") as f:
                saved_weights = pickle.load(f)
            return [Agent(i, base_weights=w) for i, w in enumerate(saved_weights)]
        else:
            print("No checkpoint found. Initializing Gen 0...")
            return [Agent(i) for i in range(NUM_AGENTS)]

    def save_population(self):
        weights = [a.base_weights for a in self.agents]
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(weights, f)

    def generate_world_geometry(self, preserve_agents=False):
        reserved_agent_tiles = set()
        if preserve_agents:
            reserved_agent_tiles = {(a.x, a.y) for a in self.agents if a.alive}

        all_coords = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in reserved_agent_tiles
        ]
        random.shuffle(all_coords)

        self.mines = all_coords[:NUM_MINES]
        self.monuments = all_coords[NUM_MINES : NUM_MINES + NUM_MONUMENTS]

        self.mine_set = set(self.mines)
        self.monument_set = set(self.monuments)
        self.all_opps = self.mines + self.monuments

        self.oasis_tiles = set()
        for mx, my in self.all_opps:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    tx, ty = mx + dx, my + dy
                    if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                        if (tx, ty) not in self.mine_set and (
                            tx,
                            ty,
                        ) not in self.monument_set:
                            self.oasis_tiles.add((tx, ty))

        self.opportunities_array = np.array(self.all_opps)
        self.grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for mx, my in self.mines:
            self.grid_map[mx, my] = 1.0
        for wx, wy in self.monuments:
            self.grid_map[wx, wy] = 0.5
        for ox, oy in self.oasis_tiles:
            self.grid_map[ox, oy] = -0.5

        self.padded_grid = np.pad(
            self.grid_map, 2, mode="constant", constant_values=0.0
        )

        # Precompute mine proximity levels for zero-overhead loop processing
        self.mine_proximity_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        for mx, my in self.mines:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    tx, ty = mx + dx, my + dy
                    if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                        self.mine_proximity_grid[tx, ty] += 1

        # Spatial Map Generation
        self.closest_10_map.clear()
        if len(self.opportunities_array) > 0:
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    pos = np.array([x, y])
                    distances = np.sum(np.abs(self.opportunities_array - pos), axis=1)
                    k = min(10, len(distances))
                    idx = np.argpartition(distances, k - 1)[:k]
                    idx = idx[np.argsort(distances[idx])]
                    self.closest_10_map[(x, y)] = self.opportunities_array[idx]
        else:
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    self.closest_10_map[(x, y)] = []

    def reposition_agents(self):
        reserved = set(self.all_opps)
        for agent in self.agents:
            while True:
                rx = random.randint(0, GRID_SIZE - 1)
                ry = random.randint(0, GRID_SIZE - 1)
                if (rx, ry) not in reserved:
                    agent.x, agent.y = rx, ry
                    agent.prev_x, agent.prev_y = rx, ry
                    break

    def compile_sensory_rune(self, agent, closest_10):
        sensory_vector = [
            agent.x / GRID_SIZE,
            agent.y / GRID_SIZE,
            agent.prev_x / GRID_SIZE,
            agent.prev_y / GRID_SIZE,
            agent.health / 200.0,
            agent.prev_health / 200.0,
            agent.wealth / 100.0,
            agent.prev_wealth / 100.0,
        ]

        for i in range(10):
            if i < len(closest_10):
                ox, oy = closest_10[i]
                sensory_vector.extend([ox / GRID_SIZE, oy / GRID_SIZE])
                if abs(agent.x - ox) <= 2 and abs(agent.y - oy) <= 2:
                    val = 1.0 if (ox, oy) in self.mine_set else 0.5
                    sensory_vector.append(val)
                else:
                    sensory_vector.append(0.0)
            else:
                sensory_vector.extend([0.0, 0.0, 0.0])

        px, py = agent.x + 2, agent.y + 2
        local_vision = self.padded_grid[px - 2 : px + 3, py - 2 : py + 3].flatten()
        sensory_vector.extend(local_vision)

        if "target" in agent.memory:
            tx, ty = agent.memory["target"]
            sensory_vector.extend([tx / GRID_SIZE, ty / GRID_SIZE])
        else:
            sensory_vector.extend([-1.0, -1.0])

        return np.array(sensory_vector, dtype=np.float32)

    def step_tick(self):
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return False

        current_positions = {(a.x, a.y): a for a in alive_agents}

        for agent in alive_agents:
            closest_10 = self.closest_10_map[(agent.x, agent.y)]

            # Index 0 is structurally identical to running min() across all targets
            closest_before = closest_10[0]
            dist_before = abs(closest_before[0] - agent.x) + abs(
                closest_before[1] - agent.y
            )

            sensory_vector = self.compile_sensory_rune(agent, closest_10)
            mov_act, mem_act = agent.forward_and_adapt(sensory_vector)

            if mem_act == 1:
                agent.memory["target"] = (agent.x, agent.y)
            elif mem_act == 2:
                agent.memory.clear()

            dx, dy = 0, 0
            if mov_act == 0:
                dy = -1
            elif mov_act == 1:
                dy = 1
            elif mov_act == 2:
                dx = 1
            elif mov_act == 3:
                dx = -1

            nx, ny = agent.x + dx, agent.y + dy
            move_valid = True

            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                move_valid = False
            elif (nx, ny) in self.mine_set or (nx, ny) in self.monument_set:
                move_valid = False
            elif (nx, ny) in current_positions and (nx, ny) != (agent.x, agent.y):
                move_valid = False

            if move_valid:
                if (agent.x, agent.y) in current_positions:
                    del current_positions[(agent.x, agent.y)]
                agent.x, agent.y = nx, ny
                current_positions[(agent.x, agent.y)] = agent

                dist_after = abs(closest_before[0] - agent.x) + abs(
                    closest_before[1] - agent.y
                )
                if dist_after < dist_before:
                    agent.fitness += 1.5

                if "target" in agent.memory:
                    tx, ty = agent.memory["target"]
                    m_dist_before = abs(tx - (nx - dx)) + abs(ty - (ny - dy))
                    m_dist_after = abs(tx - nx) + abs(ty - ny)
                    if m_dist_after < m_dist_before:
                        agent.fitness += 1.0

            if (agent.x, agent.y) in self.oasis_tiles:
                agent.health -= 1
            else:
                agent.health -= 2

            # O(1) grid value pulls replace the 9-tile search loop entirely
            is_near_mine = self.mine_proximity_grid[agent.x, agent.y]

            if is_near_mine > 0:
                agent.wealth += 2 * is_near_mine
                agent.fitness += 5.0
                while agent.wealth >= 4 and agent.health < 200:
                    agent.wealth -= 4
                    agent.health += 1

            if agent.health <= 0:
                agent.alive = False
            else:
                agent.ticks_survived += 1
                agent.fitness += 1.0

            agent.prev_x = agent.x
            agent.prev_y = agent.y
            agent.prev_health = agent.health
            agent.prev_wealth = agent.wealth

        return True

    def render_pygame(self, screen, font, generation, tick):
        BG_COLOR = (15, 23, 42)
        GRID_COLOR = (30, 41, 59)
        OASIS_COLOR = (6, 182, 212)
        MINE_COLOR = (239, 68, 68)
        MONUMENT_COLOR = (234, 179, 8)
        HARVEST_LINE_COLOR = (239, 68, 68, 100)

        screen.fill(BG_COLOR)

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        for ox, oy in self.oasis_tiles:
            rect = pygame.Rect(
                ox * CELL_SIZE + 2, oy * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4
            )
            pygame.draw.rect(screen, OASIS_COLOR, rect, border_radius=4)

        for mx, my in self.mines:
            cx = mx * CELL_SIZE + CELL_SIZE // 2
            cy = my * CELL_SIZE + CELL_SIZE // 2
            offset = CELL_SIZE // 2 - 4
            pygame.draw.polygon(
                screen,
                MINE_COLOR,
                [
                    (cx, cy - offset),
                    (cx + offset, cy),
                    (cx, cy + offset),
                    (cx - offset, cy),
                ],
            )

        for wx, wy in self.monuments:
            cx = wx * CELL_SIZE + CELL_SIZE // 2
            cy = wy * CELL_SIZE + CELL_SIZE // 2
            offset = CELL_SIZE // 2 - 4
            pygame.draw.polygon(
                screen,
                MONUMENT_COLOR,
                [
                    (cx, cy - offset),
                    (cx + offset, cy + offset),
                    (cx - offset, cy + offset),
                ],
            )

        alive_count = 0
        avg_health = 0
        for a in self.agents:
            if a.alive:
                alive_count += 1
                avg_health += a.health
                color = (
                    (34, 197, 94)
                    if a.health > 140
                    else (234, 179, 8) if a.health > 70 else (239, 68, 68)
                )
                agent_cx = a.x * CELL_SIZE + CELL_SIZE // 2
                agent_cy = a.y * CELL_SIZE + CELL_SIZE // 2

                for mx, my in self.mines:
                    if abs(a.x - mx) <= 1 and abs(a.y - my) <= 1:
                        mine_cx = mx * CELL_SIZE + CELL_SIZE // 2
                        mine_cy = my * CELL_SIZE + CELL_SIZE // 2
                        pygame.draw.line(
                            screen,
                            HARVEST_LINE_COLOR,
                            (agent_cx, agent_cy),
                            (mine_cx, mine_cy),
                            2,
                        )

                pygame.draw.circle(
                    screen, color, (agent_cx, agent_cy), CELL_SIZE // 2 - 4
                )

        avg_health = avg_health / max(1, alive_count)

        ui_panel = pygame.Surface((WINDOW_SIZE, 40))
        ui_panel.set_alpha(200)
        ui_panel.fill((0, 0, 0))
        screen.blit(ui_panel, (0, 0))

        status_text = f"GEN: {generation:02d} | TICK: {tick:03d} | ALIVE: {alive_count:02d}/{NUM_AGENTS} | AVG HP: {avg_health:.1f}"
        text_surface = font.render(status_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

        if (
            tick > 0
            and tick % SHUFFLE_INTERVAL <= (MAX_TICKS_PER_GEN // SHUFFLE_INTERVAL)
            and tick < MAX_TICKS_PER_GEN - SHUFFLE_INTERVAL
        ):
            alert_font = pygame.font.SysFont("consolas", 24, bold=True)
            alert_surface = alert_font.render(
                "⚠ MAP RE-SHUFFLE EXECUTED ⚠", True, (239, 68, 68)
            )
            screen.blit(
                alert_surface,
                (WINDOW_SIZE // 2 - alert_surface.get_width() // 2, WINDOW_SIZE // 2),
            )

        pygame.display.flip()

    def run_evolutionary_filter(self, current_gen):
        self.agents.sort(key=lambda a: a.fitness, reverse=True)

        top_slice_size = int(NUM_AGENTS * 0.25)
        survivors = self.agents[:top_slice_size]

        max_fit = survivors[0].fitness
        avg_fit = sum(a.fitness for a in self.agents) / NUM_AGENTS
        avg_ticks = sum(a.ticks_survived for a in self.agents) / NUM_AGENTS
        max_ticks = max(a.ticks_survived for a in self.agents)
        avg_wealth = sum(a.wealth for a in self.agents) / NUM_AGENTS

        next_generation = []
        for index, parent in enumerate(survivors):
            partner = random.choice(survivors)

            base_blueprint = {k: v.copy() for k, v in parent.base_weights.items()}
            agent_base = Agent(
                agent_id=len(next_generation), base_weights=base_blueprint
            )
            next_generation.append(agent_base)

            lived_blueprint = {k: v.copy() for k, v in parent.weights.items()}
            agent_lived = Agent(
                agent_id=len(next_generation), base_weights=lived_blueprint
            )
            next_generation.append(agent_lived)

            crossed_mutated_base = {}
            for layer_key, matrix_A in parent.base_weights.items():
                matrix_B = partner.base_weights[layer_key]
                mask = np.random.rand(*matrix_A.shape) > 0.5
                crossed = np.where(mask, matrix_A, matrix_B)
                noise = np.random.randn(*crossed.shape) * MUTATION_RATE
                crossed_mutated_base[layer_key] = crossed + noise
            agent_mut_base = Agent(
                agent_id=len(next_generation), base_weights=crossed_mutated_base
            )
            next_generation.append(agent_mut_base)

            crossed_mutated_lived = {}
            for layer_key, matrix_A in parent.weights.items():
                matrix_B = partner.weights[layer_key]
                mask = np.random.rand(*matrix_A.shape) > 0.5
                crossed = np.where(mask, matrix_A, matrix_B)
                noise = np.random.randn(*crossed.shape) * MUTATION_RATE
                crossed_mutated_lived[layer_key] = crossed + noise
            agent_mut_lived = Agent(
                agent_id=len(next_generation), base_weights=crossed_mutated_lived
            )
            next_generation.append(agent_mut_lived)

        self.agents = next_generation
        self.generate_world_geometry()
        self.reposition_agents()

        save_interval = max(1, NUM_GENERATIONS // 100)
        if current_gen % save_interval == 0 or current_gen == NUM_GENERATIONS - 1:
            self.save_population()

        return max_fit, avg_fit, avg_ticks, max_ticks, avg_wealth


def plot_performance(stats):
    if not stats:
        return
    gens = [s[0] for s in stats]
    max_fit = [s[1] for s in stats]
    avg_fit = [s[2] for s in stats]
    avg_life = [s[3] for s in stats]
    max_life = [s[4] for s in stats]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color=color)
    ax1.plot(gens, max_fit, label="Max Fitness", color="dodgerblue", marker="o")
    ax1.plot(gens, avg_fit, label="Avg Fitness", color="navy", linestyle="--")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Max Lifespan (Ticks)", color=color)
    ax2.plot(gens, max_life, label="Max Lifespan", color=color, marker="x")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="lower right")

    plt.title("Trinitarian Brain Ecosystem: Evolutionary Performance")
    fig.tight_layout()
    plt.show()


# --- RUNTIME EXECUTION GATEWAY ---
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    if SHOW_ON:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Trinitarian Ecosystem V4.0")
        font = pygame.font.SysFont("consolas", 18, bold=True)
        clock = pygame.time.Clock()
        print("\nLaunching Pygame Graphical Engine...")
    else:
        print("\nLaunching Headless Simulation Engine (Max Speed)...")
    ecosystem = EvolutionaryEcosystem()
    stats_history = []

    for gen in range(NUM_GENERATIONS):
        tick = 0
        start_time = time.time()
        NUM_MINES = min(
            MAX_NUM_MINES,
            max(
                MIN_NUM_MINES,
                round(
                    MAX_NUM_MINES
                    - 0.1 * DIFFICULTY_CURVE ** (gen * np.sin(2 * np.pi * gen / 1000))
                ),
            ),
        )
        NUM_MONUMENTS = min(
            MAX_NUM_MONUMENTS,
            max(
                MIN_NUM_MONUMENTS,
                round(
                    MAX_NUM_MONUMENTS
                    - 0.1 * DIFFICULTY_CURVE ** (gen * np.sin(2 * np.pi * gen / 1000))
                ),
            ),
        )

        while tick < MAX_TICKS_PER_GEN:
            if SHOW_ON:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            if tick > 0 and tick % SHUFFLE_INTERVAL == 0:
                ecosystem.generate_world_geometry(preserve_agents=True)

            active = ecosystem.step_tick()
            if not active:
                break

            if SHOW_ON:
                ecosystem.render_pygame(screen, font, gen, tick)
                clock.tick(FPS)

            tick += 1

        MUTATION_RATE *= 1 - 2 / NUM_GENERATIONS

        elapsed_time = time.time() - start_time
        tps = tick / elapsed_time if elapsed_time > 0 else 0

        mx_f, av_f, av_t, mx_t, av_w = ecosystem.run_evolutionary_filter(gen)
        stats_history.append((gen, mx_f, av_f, av_t, mx_t, av_w))

        if not SHOW_ON:
            print(
                f"GEN: {gen:03d} | MAX TICKS: {mx_t:<4} | MAX FIT: {mx_f:<8.1f}|AVG FIT: {av_f:<8.1f}|AVG TICKS: {av_t:<8.1f} | TPS: {tps:,.0f}/s"
            )

    if SHOW_ON:
        pygame.quit()

    print("\n" + "=" * 50)
    print(" SIMULATION SUCCESSFUL - HISTORICAL GENETIC STATS ")
    print("=" * 50)
    print(
        f"{'GEN':<5}{'MAX FITNESS':<15}{'AVG FITNESS':<15}{'MAX LIFESPAN':<15}{'AVG WEALTH':<10}"
    )
    print("-" * 65)
    for g, mf, af, at, mt, aw in stats_history:
        print(f"{g:<5}{mf:<15.2f}{af:<15.2f}{at:<15.2f}{mt:<15}{aw:<10.2f}")
    print("=" * 50)

    print("\nGenerating Analytics Visualization...")
    plot_performance(stats_history)
