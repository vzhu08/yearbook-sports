import os
import glob
from src.text_extraction import extract_yearbook_ocr


def main():

    # Find all PDF files in the input_pdfs folder
    input_pattern = os.path.join("pdf_input", "*.pdf")
    all_pdfs = glob.glob(input_pattern)

    if not all_pdfs:
        print("No PDF files found in 'input_pdfs' directory.")
        return

    # Process each yearbook PDF
    for pdf_path in all_pdfs:
        # Derive an output directory per PDF
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join("output", base)
        print(f"[START] Processing {pdf_path} -> {output_dir}")
        extract_yearbook_ocr(
            pdf_path=pdf_path,
            output_dir=output_dir,
            year=1980,
            name_correction_enabled=False,
        )

    print("Pipeline complete for all yearbooks.")


if __name__ == "__main__":
    main()
