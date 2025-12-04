"""Example how to save a figure as pdf file with logo."""
import numpy as np
from matplotlib import pyplot as plt

from stereographic_projection.helpers.pdf_helpers.figure2pdf import save_figure


def example_with_local_logo():
    """Example with local logo file."""
    # Create polar scatter plot
    fig, ax = _create_polar_scatter()

    # Footer text
    footer = "© 2025 AstraGeek. All rights reserved."

    save_figure(
        fig=fig,
        filename="polar_scatter_local_logo.pdf",
        logo_path="logo_astrageek.png",
        footer_text=footer,
        logo_position=(0.15, 0.97),
        text_position=(0.5, 0.01),
    )

    plt.show()

def _create_polar_scatter():
    """Create a scatter plot in polar coordinates."""
    # Set up the figure with polar projection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')

    # Generate sample data
    np.random.seed(42)
    n_points = 50
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    radius = np.random.uniform(0, 5, n_points)
    colors = np.random.rand(n_points)
    sizes = np.random.uniform(20, 200, n_points)

    ax.scatter(
        theta,
        radius,
        c=colors,
        s=sizes,
        alpha=0.7,
        cmap='viridis',
        edgecolors='white',
        linewidth=0.5,
    )

    ax.set_title("Polar Scatter Plot", va='bottom', fontsize=14, pad=20)
    ax.set_xlabel("Angle (θ)", labelpad=15)

    return fig, ax


if __name__ == "__main__":
    example_with_local_logo()
