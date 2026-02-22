# Secure Logs

Setup and installation guide for the Secure Logs project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment support

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd secure-logs
```

### 2. Create and activate virtual environment

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


### 4. Verify installation
```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

- `main.py` - Entry point
- `models/` - Model implementations
- `training/` - Training scripts
- `inference/` - Inference modules
- `retrieval/` - Retrieval components
- `config/` - Configuration files
- `utils/` - Utility functions

## Deactivating the Virtual Environment

When you're done working:
```bash
deactivate
```