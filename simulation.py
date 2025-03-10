import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

class FollowCarSimulation:
    """
    This class simulates a follow car interaction where multiple follower cars follow a lead car.
    Sliders allow dynamically adjusting the number of follower cars and the safe distance threshold.
    """
    def __init__(self, num_followers=3, safe_distance=1.0, follower_speed=0.05, lead_speed=0.03, repulsion_strength=0.1, area_size=10):
        self.num_followers = num_followers
        self.safe_distance = safe_distance
        self.follower_speed = follower_speed
        self.lead_speed = lead_speed
        self.repulsion_strength = repulsion_strength
        self.area_size = area_size

        # Car positions: followers are stored in a (num_followers, 2) array and lead car as a 2D point.
        self.followers = None  
        self.lead_car = None  

        # Matplotlib patches to represent cars.
        self.follower_patches = []
        self.lead_patch = None

        # Create figure and axis.
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.3)

        # Initialize sliders (configured later)
        self.slider_num_followers = None
        self.slider_safe_distance = None

        self._initialize_positions()
        self._setup_plot()
        self._setup_sliders()

    def _initialize_positions(self):
        """Randomly initialize positions for followers and lead car within the simulation area."""
        self.followers = np.random.rand(self.num_followers, 2) * self.area_size
        self.lead_car = np.random.rand(2) * self.area_size

    def create_car(self, position, direction, color):
        """
        Create a matplotlib.Polygon representing a car as a triangle.
        
        Args:
            position (np.array): 2D coordinates of the car.
            direction (np.array): 2D vector indicating the car's facing.
            color (str): Color string for the car.
            
        Returns:
            plt.Polygon: A polygon patch representing the car.
        """
        norm = np.linalg.norm(direction)
        if norm == 0:
            unit_direction = np.array([1, 0])
        else:
            unit_direction = direction / norm
        # Base triangle shape.
        triangle = np.array([[0, 0], [-0.2, -0.1], [-0.2, 0.1]])
        # Create a 2D rotation matrix.
        rotation_matrix = np.array([[unit_direction[0], -unit_direction[1]], [unit_direction[1], unit_direction[0]]])
        rotated_triangle = triangle.dot(rotation_matrix)
        rotated_triangle += position
        return plt.Polygon(rotated_triangle, closed=True, color=color)

    def _setup_plot(self):
        """Initialize the plot with car patches."""
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        # Create follower patches.
        self.follower_patches = []
        for i in range(self.num_followers):
            patch = self.create_car(self.followers[i], np.array([1, 1]), 'blue')
            self.follower_patches.append(patch)
            self.ax.add_patch(patch)
        # Create lead patch.
        self.lead_patch = self.create_car(self.lead_car, np.array([1, 1]), 'red')
        self.ax.add_patch(self.lead_patch)

    def _setup_sliders(self):
        """Set up sliders to control the number of followers and safe distance threshold."""
        ax_num_followers = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_safe_distance = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_num_followers = Slider(ax_num_followers, 'Number of Followers', 1, 5, valinit=self.num_followers, valstep=1)
        self.slider_safe_distance = Slider(ax_safe_distance, 'Safe Distance', 0.5, 5.0, valinit=self.safe_distance)

    def update(self, frame):
        """
        Update function called for each frame of the animation.
        It updates positions of cars, handles slider changes, and updates the visualization.
        """
        # Update number of followers if slider has been modified.
        new_num_followers = int(self.slider_num_followers.val)
        if new_num_followers != self.num_followers:
            self.num_followers = new_num_followers
            self._reinitialize_cars()

        # Update safe distance from slider.
        self.safe_distance = self.slider_safe_distance.val

        # Lead car behavior: Move along a predefined path.
        self.lead_car += np.array([self.lead_speed, 0])
        self.lead_car = np.mod(self.lead_car, self.area_size)  # Wrap around the area.

        # Follower car behavior: Move toward the lead car and apply repulsion if too close to another follower.
        for i in range(self.num_followers):
            direction_to_lead = self.lead_car - self.followers[i]
            norm_dir = np.linalg.norm(direction_to_lead)
            if norm_dir != 0:
                self.followers[i] += (direction_to_lead / norm_dir) * self.follower_speed
            # Avoid collisions with other followers.
            for j in range(self.num_followers):
                if i != j:
                    diff = self.followers[i] - self.followers[j]
                    distance = np.linalg.norm(diff)
                    if distance < 1.0 and distance != 0:
                        self.followers[i] += (diff / distance) * self.repulsion_strength

        # Update follower patches.
        for i in range(self.num_followers):
            car_direction = self.lead_car - self.followers[i]
            norm_car = np.linalg.norm(car_direction)
            if norm_car == 0:
                car_direction = np.array([1, 0])
            else:
                car_direction = car_direction / norm_car
            new_poly = self.create_car(self.followers[i], car_direction, 'blue')
            self.follower_patches[i].set_xy(new_poly.get_xy())

        # Update lead patch ▋
