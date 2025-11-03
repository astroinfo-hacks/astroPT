# Activate the environment at CC

## unset PYTHONPATH and LD_LIBRARY_PATH to avoid conflicts
unset PYTHONPATH
unset LD_LIBRARY_PATH

# Activate the conda environment
# test if /opt/conda/etc/profile.d/conda.sh exists
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
else
    # use module to load conda
    module load conda
fi

conda activate /pbs/throng/training/astroinfo2025/env/astroPT

unset PYTHONPATH
export PYTHONPATH=/pbs/throng/training/astroinfo2025/env/astroPT/lib/python3.11/site-packages:$PYTHONPATH