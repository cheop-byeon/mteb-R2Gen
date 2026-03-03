# Dataset Download & Path Resolution Usage

## CodeConvo Dataset

### Basic Download
```bash
# Download entire CodeConvo dataset
python download_CodeConvo.py
```

### Download Whole Repository with `huggingface-cli`
```bash
# Install CLI (if needed)
pip install -U "huggingface_hub[cli]"

# Download full CodeConvo repository to local folder
huggingface-cli download jiebi/CodeConvo --repo-type dataset --local-dir ./dataset/CodeConvo
```

### Path Resolution Examples

**Train split** (structure: `train/{direction}/`)
```bash
# Get train/c2i folder (default direction)
python download_CodeConvo.py --split train

# Get train/i2c folder
python download_CodeConvo.py --split train --direction i2c
```

**Dev/Test splits** (structure: `{repo}/{direction}/{split}/`)
```bash
# Get test data for specific repo and direction
python download_CodeConvo.py --split test --repo ids --direction c2i

# Get dev data for kubernetes repo (i2c direction)
python download_CodeConvo.py --split dev --repo kubernetes --direction i2c

# Other repos: ids, ids-supp, swe, kubernetes
python download_CodeConvo.py --split test --repo swe --direction i2c
```

**Path resolution without downloading**
```bash
# Only resolve path, skip download
python download_CodeConvo.py --split train --no-download
python download_CodeConvo.py --split test --repo ids --direction c2i --no-download
```

### Notes
- For `--split train`: `--direction` is optional (defaults to `c2i`)
- For `--split dev/test`: Both `--repo` and `--direction` are **required**
- Valid directions: `c2i`, `i2c`
- Valid repos: `ids`, `ids-supp`, `swe`, `kubernetes`

---

## RFCAlign Dataset

### Basic Download
```bash
# Download entire RFCAlign dataset
python download_RFCAlign.py
```

### Download Whole Repository with `huggingface-cli`
```bash
# Install CLI (if needed)
pip install -U "huggingface_hub[cli]"

# Download full RFCAlign repository to local folder
huggingface-cli download jiebi/RFCAlign --repo-type dataset --local-dir ./dataset/RFCAlign
```

### Parameter Options

```bash
python download_RFCAlign.py [--split <value>] [--topic <value>] [--no-download]
```

#### `--split`
Top-level folder to download.

Allowed values:
- `llama_non-verbose`
- `llama_verbose`
- `qwen_non-verbose`
- `qwen_verbose`

#### `--topic`
Optional topic file name **without** `.jsonl`.

Rules:
- Must be used together with `--split`
- Downloads only one file: `<split>/<topic>.jsonl`

Examples:
- `--topic ace`
- `--topic quic`
- `--topic tls`

#### `--no-download`
Inspect remote repository structure and local folder tree only.
No files are downloaded.

### Usage Examples

```bash
# 1) Inspect only (no download)
python download_RFCAlign.py --no-download

# 2) Download full RFCAlign dataset
python download_RFCAlign.py

# 3) Download one split folder only
python download_RFCAlign.py --split qwen_verbose

# 4) Download one specific file only
python download_RFCAlign.py --split qwen_verbose --topic ace

# 5) Another one-file download example
python download_RFCAlign.py --split llama_non-verbose --topic tls
```

### Notes
- Download target directory: `./dataset/RFCAlign/`
- Full download: all available files under all split folders
- Split download: only files under selected split
- Split + topic download: only one `.jsonl` file
- Files are downloaded with actual content (not fake 1B pointers)