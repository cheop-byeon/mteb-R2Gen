# Dataset Download & Path Resolution Usage

## CodeConvo Dataset

### Basic Download
```bash
# Download entire CodeConvo dataset
python download_CodeConvo.py
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

### Notes
- RFCAlign downloads the complete dataset to `./dataset/RFCAlign/`
- No path resolution options (entire dataset is downloaded at once)
- Structure includes verbose/non-verbose variants of query/passage pairs