# LST Cluster: Ollama + Gemma4 GPU Smoke Test

This guide records how to run `tests/ollama_gemma4_test.py` on the **LST / COLI** HTCondor cluster with a **Tony GPU**.

- **Part 1** — first-time setup (no Python/Ollama yet) through the first successful job  
- **Part 2** — re-run the same smoke test after everything is already installed  

Replace `rbahceci` with your username if different.

---

## Mental model

| Machine | Role |
|---------|------|
| Your PC | Edit code; `scp` files if needed |
| `login.coli.uni-saarland.de` | SSH entry; file setup; **no** `condor_submit` |
| `submit.coli.uni-saarland.de` | Submit and monitor Condor jobs (`condor_submit`, `condor_q`) |
| GPU worker (e.g. `tony-2`) | Where the job actually runs Ollama + Python |

**Important**

- Conda/venv = Python + `pip install ollama` (client only).  
- Ollama **server** = separate Linux binary under nethome.  
- Docker is **not** required for this smoke test (`universe = vanilla`).  
- Do **not** use `accounting_group = pausable` unless LST has enabled it for your account.

---

## Layout used in this guide

```text
/nethome/rbahceci/
  miniconda3/                          # Miniconda (base Python)
  gemma4_test/
    ollama_gemma4_test.py              # smoke script
    .venv/                             # Python venv (client + deps)
    ollama_bin/                        # Ollama server binary (after first download)
      bin/ollama
    cluster/
      rename_gpus.sh
      run.sh
      job.sub
    logs/                              # optional local notes

/scratch/rbahceci/
  gemma4_test/                         # Ollama models + serve log (job scratch)
  logs/gemma4_test/logfiles/           # Condor .out / .err / .log
```

---

# PART 1 — From zero to first successful GPU run

## 0. Log in

From your PC (university network or VPN as required):

```bash
ssh rbahceci@login.coli.uni-saarland.de
```

Check:

```bash
whoami
hostname
# expect: login.coli.uni-saarland.de
```

You will move to the **submit** host later for `condor_submit`. You can do most file setup on login (nethome is shared).

---

## 1. Create project folder and smoke script

```bash
mkdir -p /nethome/rbahceci/gemma4_test/cluster
mkdir -p /scratch/rbahceci/logs/gemma4_test/logfiles
mkdir -p /scratch/rbahceci/gemma4_test
```

If `nano` is missing, create the script with `cat`:

```bash
cat > /nethome/rbahceci/gemma4_test/ollama_gemma4_test.py << 'EOF'
"""Simple smoke test: verify Ollama gemma4 responds."""

from ollama import chat

response = chat(
    model="gemma4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.message.content)
EOF
```

Or copy from your laptop (run this in **PowerShell on Windows**, not inside SSH):

```powershell
scp "c:\Users\BetülBahçeci\Desktop\multi-turn-attack-defenses\tests\ollama_gemma4_test.py" `
  rbahceci@login.coli.uni-saarland.de:/nethome/rbahceci/gemma4_test/
```

---

## 2. Install Miniconda (once)

On the cluster (login is fine):

```bash
cd /nethome/rbahceci
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p /nethome/rbahceci/miniconda3
source /nethome/rbahceci/miniconda3/etc/profile.d/conda.sh
```

If Conda asks you to accept Terms of Service:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**Note:** Creating a full `conda create -n ...` env on `login` may get **Killed** (memory limit). Prefer a lightweight **venv** from Miniconda’s Python (next step).

---

## 3. Create a venv and install the Ollama Python client

```bash
/nethome/rbahceci/miniconda3/bin/python -V
/nethome/rbahceci/miniconda3/bin/python -c "print('ok')"

/nethome/rbahceci/miniconda3/bin/python -m venv /nethome/rbahceci/gemma4_test/.venv
source /nethome/rbahceci/gemma4_test/.venv/bin/activate

python -m pip install -U pip
python -m pip install "ollama>=0.6.0" zstandard

python -c "import ollama, zstandard; print('ok')"
```

- `ollama` (pip) = HTTP client used by the Python script  
- `zstandard` = needed to unpack the Ollama server `.tar.zst` archive  

---

## 4. Condor helper scripts

### 4a. `rename_gpus.sh` (LST UUID → CUDA index `0`)

```bash
cat > /nethome/rbahceci/gemma4_test/cluster/rename_gpus.sh << 'EOF'
#!/bin/bash
new_devices=""
IFS=',' read -ra my_array <<< "$CUDA_VISIBLE_DEVICES"
for id in "${my_array[@]}";
do
    new_devices=${new_devices}`nvidia-smi -L | grep "$id" | sed -E "s/^GPU ([0-9]+):.*$/\1/"`,
done
export CUDA_VISIBLE_DEVICES=${new_devices%?}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
EOF
```

### 4b. `run.sh` (job entrypoint)

On first run this downloads the Ollama **server** (~1.4 GB) into nethome, starts it, pulls `gemma4`, runs the smoke script.

```bash
cat > /nethome/rbahceci/gemma4_test/cluster/run.sh << 'EOF'
#!/bin/bash
set -eux

if [ -f /nethome/rbahceci/gemma4_test/cluster/rename_gpus.sh ]; then
  source /nethome/rbahceci/gemma4_test/cluster/rename_gpus.sh
fi

echo "Running on node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

source /nethome/rbahceci/gemma4_test/.venv/bin/activate

export SAVE=/scratch/rbahceci/gemma4_test
mkdir -p "$SAVE"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$SAVE/ollama_models"
mkdir -p "$OLLAMA_MODELS"

OLLAMA_DIR=/nethome/rbahceci/gemma4_test/ollama_bin
mkdir -p "$OLLAMA_DIR"

# Download Ollama server once (no sudo; current asset is .tar.zst)
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  python - << 'PY'
import urllib.request, tarfile, io
from pathlib import Path
import zstandard as zstd

url = "https://github.com/ollama/ollama/releases/download/v0.32.9/ollama-linux-amd64.tar.zst"
dst = Path("/nethome/rbahceci/gemma4_test/ollama_bin")
dst.mkdir(parents=True, exist_ok=True)
print("downloading ollama (~1.4GB)...")
data = urllib.request.urlopen(url, timeout=1800).read()
print("bytes:", len(data))
dctx = zstd.ZstdDecompressor()
tar_bytes = dctx.decompress(data, max_output_size=4_000_000_000)
with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
    tf.extractall(dst)
print("extracted to", dst)
PY
fi

export PATH="$OLLAMA_DIR/bin:$PATH"
command -v ollama
ollama --version || true

ollama serve > "$SAVE/ollama_serve.log" 2>&1 &
OLLAMA_PID=$!
sleep 15

ollama pull gemma4
python /nethome/rbahceci/gemma4_test/ollama_gemma4_test.py

kill $OLLAMA_PID || true
EOF

chmod +x /nethome/rbahceci/gemma4_test/cluster/*.sh
```

### 4c. `job.sub` (HTCondor submit description)

```bash
cat > /nethome/rbahceci/gemma4_test/cluster/job.sub << 'EOF'
universe           = vanilla
initialdir         = /nethome/rbahceci
executable         = /nethome/rbahceci/gemma4_test/cluster/run.sh

output             = /scratch/rbahceci/logs/gemma4_test/logfiles/run.$(Cluster)_$(Process).out
error              = /scratch/rbahceci/logs/gemma4_test/logfiles/run.$(Cluster)_$(Process).err
log                = /scratch/rbahceci/logs/gemma4_test/logfiles/run.$(Cluster)_$(Process).log

request_CPUs       = 2
request_memory     = 16G
request_GPUs       = 1
requirements       = (GPUs_GlobalMemoryMb >= 16000)

+MaxWallTime       = 7200

queue 1
EOF
```

Optional: pin to Tony GPUs only:

```text
requirements = (TARGET.UidDomain == "coli.uni-saarland.de") && \
               regexp(".*@tony-", TARGET.Name) && \
               (GPUs_GlobalMemoryMb >= 16000)
```

Do **not** add `accounting_group = pausable` unless your account is allowed (otherwise submit fails with permission denied).

---

## 5. Go to the submit host and submit

From login:

```bash
ssh submit
# same as: ssh submit.coli.uni-saarland.de

hostname
which condor_submit
# expect: /usr/bin/condor_submit
```

Submit:

```bash
mkdir -p /scratch/rbahceci/logs/gemma4_test/logfiles
cd /nethome/rbahceci/gemma4_test/cluster
condor_submit job.sub
condor_q
```

Example success line:

```text
1 job(s) submitted to cluster 105071.
```

`condor_q` may show **0 jobs** quickly if the job already finished; that can be normal.

---

## 6. Check results

```bash
ls -lt /scratch/rbahceci/logs/gemma4_test/logfiles/
condor_history rbahceci -limit 5

# replace JOBID with your cluster id, e.g. 105071
cat /scratch/rbahceci/logs/gemma4_test/logfiles/run.JOBID_0.out
cat /scratch/rbahceci/logs/gemma4_test/logfiles/run.JOBID_0.err
```

**Success looks like**

- Node such as `tony-2.coli.uni-saarland.de`
- `CUDA_VISIBLE_DEVICES=0` (after rename)
- Ollama download/extract (first time only)
- Final line similar to: `Hello! How can I help you today?`

A warning like `could not connect to a running Ollama instance` during `ollama --version` (before serve is up) can be ignored if the Python script still prints a reply.

---

## Part 1 pitfalls (already hit once)

| Problem | Fix |
|---------|-----|
| `condor_submit` not found on login | `ssh submit` first |
| `conda create` **Killed** on login | Use `python -m venv` instead |
| `sudo` Ollama install denied | User-local `.tar.zst` download in `run.sh` |
| Download URL 404 for `.tgz` | Use `ollama-linux-amd64.tar.zst` + `zstandard` |
| `accounting_group = pausable` denied | Remove that line from `.sub` |
| GPU UUID in `CUDA_VISIBLE_DEVICES` | Source `rename_gpus.sh` |

---

# PART 2 — Run the smoke test again (everything already installed)

Use this when:

- Miniconda + `.venv` exist  
- `ollama_bin/bin/ollama` already exists  
- `gemma4` may already be cached under scratch/`OLLAMA_MODELS`  
- You only want to chat with the model again via `ollama_gemma4_test.py`

## 1. Log in and go to submit

```bash
ssh rbahceci@login.coli.uni-saarland.de
ssh submit
```

## 2. Confirm pieces still exist

```bash
ls /nethome/rbahceci/gemma4_test/ollama_gemma4_test.py
ls /nethome/rbahceci/gemma4_test/.venv/bin/python
ls /nethome/rbahceci/gemma4_test/ollama_bin/bin/ollama
ls /nethome/rbahceci/gemma4_test/cluster/run.sh
ls /nethome/rbahceci/gemma4_test/cluster/job.sub
```

Optional client check (on submit/login; does not need a GPU):

```bash
source /nethome/rbahceci/gemma4_test/.venv/bin/activate
python -c "import ollama; print('ok')"
```

## 3. Submit the same job

`run.sh` skips the big download if `ollama_bin/bin/ollama` is already executable.

```bash
cd /nethome/rbahceci/gemma4_test/cluster
condor_submit job.sub
condor_q
```

## 4. Read the new output

```bash
ls -lt /scratch/rbahceci/logs/gemma4_test/logfiles/ | head
# open the newest run.*.out
tail -n 30 /scratch/rbahceci/logs/gemma4_test/logfiles/run.*_0.out
```

You should again see Gemma’s reply at the end.

## 5. Optional: change the prompt

Edit the script on the cluster, then resubmit:

```bash
# on login or submit
cat /nethome/rbahceci/gemma4_test/ollama_gemma4_test.py
# edit with cat/vi, or scp a new copy from your PC
```

Then repeat Part 2 §3–4.

## 6. Optional: interactive GPU job (advanced)

If you want a shell on a GPU node (wiki: interactive jobs + `condor_ssh_to_job`), use the LST interactive pattern, then manually:

```bash
source /nethome/rbahceci/gemma4_test/.venv/bin/activate
export PATH=/nethome/rbahceci/gemma4_test/ollama_bin/bin:$PATH
# source rename_gpus.sh if needed
ollama serve &
sleep 5
ollama run gemma4
# or: python /nethome/rbahceci/gemma4_test/ollama_gemma4_test.py
```

Batch `job.sub` is enough for the smoke test.

---

## Quick reference

```bash
# Login path
ssh rbahceci@login.coli.uni-saarland.de
ssh submit

# Submit smoke test
cd /nethome/rbahceci/gemma4_test/cluster
condor_submit job.sub
condor_q
condor_history rbahceci -limit 3
ls -lt /scratch/rbahceci/logs/gemma4_test/logfiles/

# Useful paths
# Python:  /nethome/rbahceci/gemma4_test/.venv/bin/python
# Ollama:  /nethome/rbahceci/gemma4_test/ollama_bin/bin/ollama
# Script:  /nethome/rbahceci/gemma4_test/ollama_gemma4_test.py
# Logs:    /scratch/rbahceci/logs/gemma4_test/logfiles/
```

---

## Related wiki pages (LST)

- Cluster overview: https://wiki.lst.uni-saarland.de/doku.php?id=user:cluster:cluster  
- Step-by-step submit: https://wiki.lst.uni-saarland.de/doku.php?id=user:cluster:job_script_example  
- Workflow example (Docker variant; not required for this smoke test): https://wiki.lst.uni-saarland.de/doku.php?id=user:cluster:c_example-workflow  

Remember: submit Condor jobs from **`submit.coli`**, not from **`login.coli`**.
