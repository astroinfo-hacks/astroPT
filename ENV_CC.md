# Setting up the astroPT Environment at CC

## Create the environment

No need to do this step, this is just for reference.

Activate conda on CC jupyter platform:

```bash
. "/opt/conda/etc/profile.d/conda.sh"
```

Create the conda environment with Python 3.11:

```bash
conda create --prefix /pbs/throng/training/astroinfo2025/env/04_generative_jax python=3.11 pip -y
conda activate /pbs/throng/training/astroinfo2025/env/astroPT
unset PYTHONPATH
pip install astropt ipykernel einops
```

## Activate the environment
 
To activate the astroPT environment, run:

```bash
source /pbs/throng/training/astroinfo2025/soft/astroPT/source_CC.sh
```

## Jupyter Environment Setup

- Create kernel directory
```bash
mkdir -p $HOME/.local/share/jupyter/kernels/astroPT
```

- Create the helper script
```bash
cat > $HOME/.local/share/jupyter/kernels/astroPT/jupyter-helper.sh << 'EOF'
#!/bin/bash
source /usr/share/Modules/init/bash

# Activate the conda environment
conda activate /pbs/throng/training/astroinfo2025/env/astroPT

# Clear and set PYTHONPATH
unset PYTHONPATH
export PYTHONPATH=/pbs/throng/training/astroinfo2025/env/astroPT/lib/python3.11/site-packages:$PYTHONPATH

# Launch the kernel
exec python -m ipykernel_launcher "$@"
EOF
```

- Make the script executable
```bash
chmod +x $HOME/.local/share/jupyter/kernels/astroPT/jupyter-helper.sh
```

- Create kernel configuration
```bash
cat > $HOME/.local/share/jupyter/kernels/astroPT/kernel.json << EOF
{
  "display_name": "astroPT",
  "language": "python",
  "argv": [
      "$HOME/.local/share/jupyter/kernels/astroPT/jupyter-helper.sh",
      "-f",
      "{connection_file}"
  ]
}
EOF
```
