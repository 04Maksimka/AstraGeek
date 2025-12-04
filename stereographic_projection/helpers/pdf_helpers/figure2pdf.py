"""Module with function that creates pdf from a given plt.figure."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from loguru import logger
from PIL import Image


def add_logo_to_figure(
    fig: plt.Figure,
    logo_path: str | None = None,
    position: tuple[float, float] = (0.05, 0.95),
    size: float = 0.1,
) -> None:
    """Add a logo to the figure.

    :param fig: Figure to be saved
    :param logo_path: path to the logo
    :param position: position, where to draw
    :param size: relative size for the logo
    """
    try:
        logo = Image.open(logo_path)
        # Convert to array for matplotlib
        logo_array = np.array(logo)
        # Create OffsetImage
        im = OffsetImage(logo_array, zoom=size)
        # Create AnnotationBbox
        ab = AnnotationBbox(
            im,
            position,
            xycoords='figure fraction',
            frameon=False,
            box_alignment=(0, 1),
        )
        # Add to figure
        fig.add_artist(ab)

    except Exception as e:
        print(f"Error loading logo: {e}")


def add_footer_text(
    fig: plt.Figure,
    text: str,
    position: tuple[float, float] = (0.5, 0.02),
    fontsize: int = 9 ,
    color: str = 'gray',
):
    """Add footer text to the figure.

    :param fig: Figure to be saved
    :param text: text to be written
    :param position: position, where to draw
    :param fontsize: fontsize of the text
    :param color: color of the text
    """
    fig.text(
        position[0], position[1], text,
        fontsize=fontsize, color=color,
        ha='center', va='bottom',
        transform=fig.transFigure,
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
    logo_path: str | None = None,
    footer_text: str | None = None,
    dpi: int = 300,
    logo_position: tuple[float, float] = (0.05, 0.95),
    text_position: tuple[float, float] = (0.5, 0.02),

):
    """Save figure as PDF with logo and footer text.

    :param filename: Name of the file where to save a fig
    :param fig: Figure to be saved
    :param logo_path: path to the logo
    :param footer_text: text to be written
    :param logo_position: position, where to draw logo
    :param text_position: position, where to draw logo
    :param dpi: dots per inch
    """
    if logo_path:
        add_logo_to_figure(fig, logo_path, logo_position)

    if footer_text:
        add_footer_text(fig, footer_text, text_position)

    if not filename.endswith('.pdf'):
        filename += '.pdf'

    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    logger.info(f"Figure saved as '{filename}'")
