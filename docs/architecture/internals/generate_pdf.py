#!/usr/bin/env python3
"""Generate PDF from mathematical_foundations.md using available Python libraries."""

import os
import sys
from pathlib import Path

import markdown


def markdown_to_html_with_css(md_file, html_file):
    """Convert markdown to HTML with proper styling for academic documents."""

    # Read markdown content
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Configure markdown with extensions for better rendering
    md = markdown.Markdown(
        extensions=[
            "extra",  # Tables, fenced code, etc.
            "codehilite",  # Code syntax highlighting
            "toc",  # Table of contents
            "tables",  # Better table support
            "fenced_code",  # Fenced code blocks
            "attr_list",  # Attribute lists
        ]
    )

    html_body = md.convert(md_content)

    # Create a complete HTML document with academic paper styling
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mathematical Foundations of Batcher</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm 2cm;
        }}

        body {{
            font-family: "Times New Roman", Times, serif;
            font-size: 11pt;
            line-height: 1.6;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
            color: #000;
            background: #fff;
        }}

        h1 {{
            font-size: 20pt;
            font-weight: bold;
            margin: 30px 0 15px 0;
            page-break-after: avoid;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}

        h2 {{
            font-size: 16pt;
            font-weight: bold;
            margin: 25px 0 12px 0;
            page-break-after: avoid;
            border-bottom: 1px solid #666;
            padding-bottom: 5px;
        }}

        h3 {{
            font-size: 13pt;
            font-weight: bold;
            margin: 20px 0 10px 0;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 11pt;
            font-weight: bold;
            margin: 15px 0 8px 0;
            page-break-after: avoid;
        }}

        p {{
            margin: 8px 0;
            text-align: justify;
        }}

        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}

        li {{
            margin: 5px 0;
        }}

        code {{
            font-family: "Courier New", Courier, monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}

        pre {{
            font-family: "Courier New", Courier, monospace;
            font-size: 9pt;
            background-color: #f8f8f8;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 12px;
            overflow-x: auto;
            margin: 15px 0;
            page-break-inside: avoid;
        }}

        pre code {{
            background: none;
            border: none;
            padding: 0;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}

        th, td {{
            border: 1px solid #999;
            padding: 8px;
            text-align: left;
        }}

        th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}

        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        blockquote {{
            margin: 15px 30px;
            padding: 10px 20px;
            border-left: 4px solid #ccc;
            background-color: #f9f9f9;
            font-style: italic;
        }}

        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
            page-break-inside: avoid;
        }}

        .math {{
            font-family: "Times New Roman", Times, serif;
            font-style: italic;
        }}

        strong {{
            font-weight: bold;
        }}

        em {{
            font-style: italic;
        }}

        hr {{
            border: none;
            border-top: 1px solid #999;
            margin: 20px 0;
        }}

        @media print {{
            body {{
                background: white;
            }}
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            table, figure, pre {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Write HTML file
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✓ Generated HTML: {html_file}")
    return html_file


def html_to_pdf_weasyprint(html_file, pdf_file):
    """Convert HTML to PDF using WeasyPrint (if available)."""
    try:
        from weasyprint import HTML

        HTML(filename=html_file).write_pdf(pdf_file)
        print(f"✓ Generated PDF using WeasyPrint: {pdf_file}")
        return True
    except Exception as e:
        print(f"✗ WeasyPrint failed: {e}")
        return False


def html_to_pdf_pdfkit(html_file, pdf_file):
    """Convert HTML to PDF using pdfkit/wkhtmltopdf (if available)."""
    try:
        import pdfkit

        pdfkit.from_file(html_file, pdf_file)
        print(f"✓ Generated PDF using pdfkit: {pdf_file}")
        return True
    except Exception as e:
        print(f"✗ pdfkit failed: {e}")
        return False


def html_to_pdf_playwright(html_file, pdf_file):
    """Convert HTML to PDF using Playwright (if available)."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{os.path.abspath(html_file)}")
            page.pdf(path=pdf_file, format="A4", print_background=True)
            browser.close()
        print(f"✓ Generated PDF using Playwright: {pdf_file}")
        return True
    except Exception as e:
        print(f"✗ Playwright failed: {e}")
        return False


def main():
    # File paths
    base_dir = Path(__file__).parent
    md_file = base_dir / "mathematical_foundations.md"
    html_file = base_dir / "mathematical_foundations.html"
    pdf_file = base_dir / "mathematical_foundations.pdf"

    if not md_file.exists():
        print(f"Error: {md_file} not found!")
        sys.exit(1)

    # Step 1: Convert markdown to HTML
    print("Step 1: Converting Markdown to HTML...")
    markdown_to_html_with_css(md_file, html_file)

    # Step 2: Convert HTML to PDF (try multiple methods)
    print("\nStep 2: Converting HTML to PDF...")

    methods = [
        ("WeasyPrint", html_to_pdf_weasyprint),
        ("pdfkit", html_to_pdf_pdfkit),
        ("Playwright", html_to_pdf_playwright),
    ]

    success = False
    for method_name, method_func in methods:
        print(f"\nTrying {method_name}...")
        if method_func(html_file, pdf_file):
            success = True
            break

    if not success:
        print("\n" + "=" * 70)
        print("⚠ Could not generate PDF automatically.")
        print("=" * 70)
        print("\nAlternative options:")
        print(f"1. Open {html_file} in a browser and use Print to PDF")
        print("2. Install pandoc: sudo apt-get install pandoc texlive-xetex")
        print("   Then run: pandoc mathematical_foundations.md -o mathematical_foundations.pdf")
        print("3. Install WeasyPrint dependencies:")
        print("   sudo apt-get install libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0")
        print(f"\nHTML file is ready at: {html_file}")
        sys.exit(1)

    # Clean up HTML file
    # os.remove(html_file)
    # print(f"\n✓ Cleaned up temporary HTML file")

    print(f"\n{'=' * 70}")
    print(f"✓ SUCCESS: PDF generated at {pdf_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
