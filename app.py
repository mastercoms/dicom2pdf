import os
import random
from pathlib import Path
from typing import Any, BinaryIO, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


def read_dicom_image(
    file_path: str | bytes | os.PathLike | BinaryIO,
) -> Tuple[np.ndarray | None, dict[str, Any] | None]:
    try:
        dicom_data = pydicom.dcmread(file_path)
        try:
            image_array = dicom_data.pixel_array
        except Exception:
            return None, None

        metadata = {
            "patient_name": getattr(dicom_data, "PatientName", "Unknown"),
            "series_description": getattr(dicom_data, "SeriesDescription", "Unknown"),
            "instance_number": getattr(dicom_data, "InstanceNumber", "Unknown"),
            "slice_location": getattr(dicom_data, "SliceLocation", "Unknown"),
            "study_date": getattr(dicom_data, "StudyDate", "Unknown"),
            "modality": getattr(dicom_data, "Modality", "Unknown"),
            "rows": getattr(dicom_data, "Rows", 0),
            "columns": getattr(dicom_data, "Columns", 0),
        }

        return image_array, metadata
    except:
        return None, None


def normalize_image(image_array: np.ndarray, contrast_factor=0.9) -> np.ndarray:
    image_array = image_array.astype(np.float64)
    p2, p98 = np.percentile(image_array, [2, 98])
    image_array = np.clip(image_array, p2, p98)
    img_min = np.min(image_array)
    img_max = np.max(image_array)
    if img_max > img_min:
        image_array = (image_array - img_min) / (img_max - img_min)
    return np.power(image_array, contrast_factor)


DICOM_EXTENSIONS = {".dcm", ".dicom", ".dic"}


def find_dicom_files(folder_path: Path):
    dicom_files = []
    for path in folder_path.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in DICOM_EXTENSIONS or not path.suffix:
                try:
                    pydicom.dcmread(str(path), stop_before_pixels=True)
                    dicom_files.append(str(path))
                except:
                    pass
    return dicom_files


def convert_to_pdf(
    dicom_files: Sequence[BinaryIO],
    output_pdf: Path,
    contrast_factor: float = 0.9,
    dpi: int = 200,
):
    if not dicom_files:
        return None

    with PdfPages(output_pdf) as pdf:
        for file in dicom_files:
            image_array, metadata = read_dicom_image(file)
            if image_array is None or metadata is None:
                continue

            norm_img = normalize_image(image_array, contrast_factor)

            fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
            ax.imshow(norm_img, cmap="gray", vmin=0, vmax=1)
            ax.set_facecolor("black")
            ax.axis("off")

            title = f"{metadata.get('patient_name', '')} | {metadata.get('series_description', '')}"
            fig.suptitle(title, fontsize=12, y=0.95, color="white")

            pdf.savefig(
                fig,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor=fig.get_facecolor(),
                dpi=dpi,
            )
            plt.close(fig)

    return output_pdf


# Streamlit app
st.set_page_config(
    page_title="DICOM to PDF Converter", page_icon=":material/scan:", layout="centered"
)
st.html(
    """
    <style>
        .main {background-color: #f5f5f5;}
        footer {visibility: hidden;}
        .custom-footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: #888;
            padding: 10px;
            background-color: #ffffff;
        }
    </style>
"""
)

st.title("DICOM to PDF Converter")
st.write(
    "Upload a folder containing your DICOM images. We'll generate a high-quality PDF scan for you."
)

dicom_files = st.file_uploader(
    "Upload DICOM folder",
    accept_multiple_files="directory",
    type=list(DICOM_EXTENSIONS),
)

contrast_factor = st.slider("Adjust Contrast", 0.5, 1.5, 0.9, step=0.05)
dpi = st.slider("Set PDF Resolution (DPI)", 100, 300, 200, step=10)

if dicom_files:
    parent = Path(dicom_files[0].name).parent
    with st.spinner("Processing your DICOM files..."):
        st.subheader("📸 Image Preview")
        preview_files = random.sample(dicom_files, min(10, len(dicom_files)))
        for file in preview_files:
            image_array, metadata = read_dicom_image(file)
            if image_array is None or metadata is None:
                continue
            norm_img = normalize_image(image_array, contrast_factor)
            st.image(
                norm_img,
                caption=str(file),
                use_column_width=True,
                clamp=True,
            )

        output_pdf_path = parent / f"{parent.name}.pdf"
        result = convert_to_pdf(
            dicom_files, output_pdf_path, contrast_factor=contrast_factor, dpi=dpi
        )

        if result and os.path.exists(output_pdf_path):
            with open(output_pdf_path, "rb") as f:
                st.success(f"✅ PDF successfully created at {output_pdf_path}!")
        else:
            st.error("Failed to generate PDF. Please check your files.")
else:
    st.info("Please upload a valid folder containing DICOM files.")
