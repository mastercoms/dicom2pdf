import os
import random
import tempfile
import traceback
from pathlib import Path
from typing import Any, BinaryIO, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

DICOM_EXTENSIONS = {".dcm", ".dicom", ".dic"}


def read_dicom_image(
    file_path: str | bytes | os.PathLike | BinaryIO,
) -> Tuple[np.ndarray | None, dict[str, str | Any] | None]:
    try:
        dicom_data = pydicom.dcmread(file_path)
        try:
            image_array = dicom_data.pixel_array
        except:
            traceback.print_exc()
            return None, None

        metadata = {
            "patient_name": getattr(dicom_data, "PatientName", ""),
            "series_description": getattr(dicom_data, "SeriesDescription", ""),
            "instance_number": getattr(dicom_data, "InstanceNumber", ""),
            "slice_location": getattr(dicom_data, "SliceLocation", ""),
            "study_date": getattr(dicom_data, "StudyDate", ""),
            "modality": getattr(dicom_data, "Modality", ""),
            "rows": getattr(dicom_data, "Rows", 0),
            "columns": getattr(dicom_data, "Columns", 0),
        }

        return image_array, metadata
    except:
        traceback.print_exc()
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


def convert_to_pdf(
    dicom_files: Sequence[BinaryIO],
    output_pdf: Path,
    contrast_factor: float = 0.9,
    dpi: int = 200,
) -> bool:
    if not dicom_files:
        return False

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

            title = metadata.get("patient_name", "")
            title = title.strip()
            title = title.replace("^", " ")
            title = title.replace("_", " ")
            if metadata.get("series_description"):
                desc = metadata["series_description"].strip()
                desc = desc.replace("^", " ")
                desc = desc.replace("_", " ")
                title += f" | {desc}"
            if title:
                fig.suptitle(title, fontsize=12, y=0.95, color="white")

            pdf.savefig(
                fig,
                bbox_inches="tight",
                pad_inches=0.1,
                facecolor=fig.get_facecolor(),
                dpi=dpi,
            )
            plt.close(fig)

    return True


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
        with tempfile.TemporaryDirectory() as tmpdir:
            st.subheader("📸 Image Preview")
            preview_limit = 10
            count = min(preview_limit, len(dicom_files))
            preview_files = (
                dicom_files
                if count <= preview_limit
                else random.sample(dicom_files, count)
            )
            if count > preview_limit:
                st.markdown(f"### Sample image previews")
            for file in preview_files:
                image_array, metadata = read_dicom_image(file)
                if image_array is None or metadata is None:
                    continue
                norm_img = normalize_image(image_array, contrast_factor)
                st.image(
                    norm_img,
                    caption=f"File: {file.name} | Data: {metadata}",
                    width="stretch",
                    clamp=True,
                )
            if count > preview_limit:
                st.info(f"And {len(dicom_files) - count} more images...")

            pdf_name = f"{parent.name}.pdf"
            output_pdf_path = Path(tmpdir) / pdf_name
            result = convert_to_pdf(
                dicom_files, output_pdf_path, contrast_factor=contrast_factor, dpi=dpi
            )

            failure_code = 0
            if not result:
                failure_code |= 1 << 0
            if not output_pdf_path.exists():
                failure_code |= 1 << 1
            if not output_pdf_path.is_file():
                failure_code |= 1 << 2
            if output_pdf_path.stat().st_size == 0:
                failure_code |= 1 << 3

            if failure_code == 0:
                with open(output_pdf_path, "rb") as f:
                    st.success("✅ PDF successfully created!")
                    st.download_button(
                        "📥 Download PDF",
                        f,
                        file_name=pdf_name,
                        mime="application/pdf",
                    )
            else:
                st.error(
                    f"Failed to generate PDF (Error code: {failure_code}). Please check your files."
                )
else:
    st.info("Please upload a valid folder containing DICOM files.")
