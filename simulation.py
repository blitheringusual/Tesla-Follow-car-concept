import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

class PredatorPreySimulation:
    """
    This class simulates a predator-prey interaction where multiple hunters chase a fleeing prey.
    Sliders allow dynamically adjusting the number of hunters and the safe distance threshold.
    """
    def __init__(self, num_agents=3, safe_distance=1.0, hunter_speed=0.05, prey_speed=0.03,
                 repulsion_strength=0.1, area_size=10):
        self.num_agents = num_agents
        self.safe_distance = safe_distance
        self.hunter_speed = hunter_speed
        self.prey_speed = prey_speed
        self.repulsion_strength = repulsion_strength
        self.area_size = area_size

        # Agent positions: hunters are stored in a (num_agents, 2) array and prey as a 2D point.
        self.hunters = None  
        self.prey = None  

        # Matplotlib patches to represent agents.
        self.hunter_patches = []
        self.prey_patch = None

        # Create figure and axis.
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.3)

        # Initialize sliders (configured later)
        self.slider_num_agents = None
        self.slider_safe_distance = None

        self._initialize_positions()
        self._setup_plot()
        self._setup_sliders()

    def _initialize_positions(self):
        """Randomly initialize positions for hunters and prey within the simulation area."""
        self.hunters = np.random.rand(self.num_agents, 2) * self.area_size
        self.prey = np.random.rand(2) * self.area_size

    def create_triangle(self, position, direction, color):
        """
        Create a matplotlib.Polygon representing an agent as a triangle.
        
        Args:
            position (np.array): 2D coordinates of the agent.
            direction (np.array): 2D vector indicating the agent's facing.
            color (str): Color string for the agent.
            
        Returns:
            plt.Polygon: A polygon patch representing the agent.
        """
        norm = np.linalg.norm(direction)
        if norm == 0:
            unit_direction = np.array([1, 0])
        else:
            unit_direction = direction / norm
        # Base triangle shape.
        triangle = np.array([[0, 0], [-0.2, -0.1], [-0.2, 0.1]])
        # Create a 2D rotation matrix.
        rotation_matrix = np.array([[unit_direction[0], -unit_direction[1]],
                                    [unit_direction[1], unit_direction[0]]])
        rotated_triangle = triangle.dot(rotation_matrix)
        rotated_triangle += position
        return plt.Polygon(rotated_triangle, closed=True, color=color)

    def _setup_plot(self):
        """Initialize the plot with agent patches."""
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        # Create hunter patches.
        self.hunter_patches = []
        for i in range(self.num_agents):
            patch = self.create_triangle(self.hunters[i], np.array([1, 1]), 'blue')
            self.hunter_patches.append(patch)
            self.ax.add_patch(patch)
        # Create prey patch.
        self.prey_patch = self.create_triangle(self.prey, np.array([1, 1]), 'red')
        self.ax.add_patch(self.prey_patch)

    def _setup_sliders(self):
        """Set up sliders to control the number of agents and safe distance threshold."""
        ax_num_agents = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_safe_distance = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_num_agents = Slider(ax_num_agents, 'Number of Agents', 1, 5,
                                        valinit=self.num_agents, valstep=1)
        self.slider_safe_distance = Slider(ax_safe_distance, 'Safe Distance', 0.5, 5.0,
                                           valinit=self.safe_distance)

    def update(self, frame):
        """
        Update function called for each frame of the animation.
        It updates positions of agents, handles slider changes, and updates the visualization.
        """
        # Update number of agents if slider has been modified.
        new_num_agents = int(self.slider_num_agents.val)
        if new_num_agents != self.num_agents:
            self.num_agents = new_num_agents
            self._reinitialize_agents()

        # Update safe distance from slider.
        self.safe_distance = self.slider_safe_distance.val

        # Prey behavior: Move away from the closest hunter.
        distances = np.linalg.norm(self.hunters - self.prey, axis=1)
        closest_idx = np.argmin(distances)
        direction_vector = self.prey - self.hunters[closest_idx]
        norm_val = np.linalg.norm(direction_vector)
        if norm_val != 0:
            self.prey += (direction_vector / norm_val) * self.prey_speed

        # Hunter behavior: Move toward the prey and apply repulsion if too close to another hunter.
        for i in range(self.num_agents):
            direction_to_prey = self.prey - self.hunters[i]
            norm_dir = np.linalg.norm(direction_to_prey)
            if norm_dir != 0:
                self.hunters[i] += (direction_to_prey / norm_dir) * self.hunter_speed
            # Avoid collisions with other hunters.
            for j in range(self.num_agents):
                if i != j:
                    diff = self.hunters[i] - self.hunters[j]
                    distance = np.linalg.norm(diff)
                    if distance < 1.0 and distance != 0:
                        self.hunters[i] += (diff / distance) * self.repulsion_strength

        # Update hunter patches.
        for i in range(self.num_agents):
            agent_direction = self.prey - self.hunters[i]
            norm_agent = np.linalg.norm(agent_direction)
            if norm_agent == 0:
                agent_direction = np.array([1, 0])
            else:
                agent_direction = agent_direction / norm_agent
            new_poly = self.create_triangle(self.hunters[i], agent_direction, 'blue')
            self.hunter_patches[i].set_xy(new_poly.get_xy())

        # Update prey patch.
        flee_direction = self.hunters[closest_idx] - self.prey
        norm_flee = np.linalg.norm(flee_direction)
        if norm_flee == 0:
            flee_direction = np.array([1, 0])
        else:
            flee_direction = flee_direction / norm_flee
        new_prey_poly = self.create_triangle(self.prey, flee_direction, 'red')
        self.prey_patch.set_xy(new_prey_poly.get_xy())

        # Check if prey is caught.
        if np.min(distances) < self.safe_distance:
            print("Prey caught!")
            plt.close()

        return self.hunter_patches + [self.prey_patch]

    def _reinitialize_agents(self):
        """Reinitialize agent positions and update plot patches after slider change."""
        # Remove all existing patches.
        for patch in self.hunter_patches:
            patch.remove()
        self.prey_patch.remove()
        # Initialize new positions.
        self._initialize_positions()
        # Re-create patches.
        self.hunter_patches = []
        for i in range(self.num_agents):
            patch = self.create_triangle(self.hunters[i], np.array([1, 1]), 'blue')
            self.hunter_patches.append(patch)
            self.ax.add_patch(patch)
        self.prey_patch = self.create_triangle(self.prey, np.array([1, 1]), 'red')
        self.ax.add_patch(self.prey_patch)

    def run(self):
        """Starts the simulation animation."""
        ani = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()

if __name__ == "__main__":
    simulation = PredatorPreySimulation()
    simulation.run()