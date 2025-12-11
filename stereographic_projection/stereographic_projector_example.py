from stereographic_projection.stereographic_projector import StereoProjector, StereoProjConfig
from stereographic_projection.hip_catalog.hip_catalog import Catalog
from stereographic_projection.helpers.pdf_helpers.figure2pdf import save_figure
import matplotlib.pyplot as plt
from datetime import datetime

def stars_with_logo(figure, need_pdf: bool = False):
    """Example with local logo file."""
    # Footer text
    footer = "© 2025 AstraGeek. All rights reserved."

    if need_pdf:
        save_figure(
            fig=figure,
            filename="polar_scatter_local_logo.pdf",
            logo_path="./helpers/pdf_helpers/logo_astrageek.png",
            footer_text=footer,
            logo_position=(0.15, 0.97),
            text_position=(0.5, 0.01),
        )

    plt.show()


if __name__ == "__main__":
    cfg = StereoProjConfig(add_ecliptic=True, utc_time=datetime(year=2021, month=3, day=1, hour=6), latitude=0, longitude=0)
    catalog = Catalog()
    proj = StereoProjector(cfg, catalog)

    figure = proj.generate()
    stars_with_logo(figure, need_pdf=False)