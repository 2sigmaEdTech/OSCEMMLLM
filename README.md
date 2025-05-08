# OSCE Video Analysis System

This project implements a multimodal AI system for analyzing OSCE (Objective Structured Clinical Examination) videos, automatically assessing performance using both binary checklists and entrustment scales.

## Project Structure

```
MMLLMs/
├── src/                    # Source code
│   ├── video_processing/   # Video processing modules
│   ├── models/            # MMLLM integration and assessment
│   ├── utils/             # Utility functions
│   └── main.py           # Main application entry
├── tests/                 # Test files
├── data/                  # Data directory
│   ├── raw_videos/       # Input OSCE videos
│   └── processed_frames/ # Processed video frames
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[To be added as development progresses]

## Features

- Video frame extraction and preprocessing
- Multimodal analysis of OSCE performances
- Automated assessment using binary checklists
- Entrustment scale scoring
- Comparison with human ratings

## Development Status

Currently in initial development phase, focusing on video processing infrastructure. 