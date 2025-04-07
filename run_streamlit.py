#!/usr/bin/env python
"""
Helper script to run the Streamlit app correctly with proper imports.
Run this script from the project root directory.
"""
import os
import sys
import subprocess


def run_streamlit():
    # Make sure we're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Add the project root to PYTHONPATH
    os.environ['PYTHONPATH'] = project_root

    # Run the Streamlit app
    streamlit_path = os.path.join(project_root, "visualization", "network_gui_streamlit.py")
    print(f"Starting Streamlit app from: {streamlit_path}")

    result = subprocess.run([
        "streamlit", "run", streamlit_path
    ], env=os.environ)

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_streamlit())