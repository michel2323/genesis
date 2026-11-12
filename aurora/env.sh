# Common environment for the Aurora job scripts. Source it from bash:  source aurora/env.sh
# Only the julia/1.12 module configures Julia/oneAPI (ONEAPI_LTS, JULIA_LOAD_PATH with the site
# driver-library preferences, NEO cache, IGC_DisableSOAPromotion, proxy, TMPDIR); do not
# override those here.

export JULIA_DEPOT_PATH=/lus/flare/projects/Julia/mschanen/julia_depot

if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
  [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi
if ! module is-avail julia/1.12 >/dev/null 2>&1; then
  module use /soft/libraries/julia/modulefiles
fi
module load julia/1.12

export ZE_AFFINITY_MASK=0              # single GPU tile (FLAT hierarchy)
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-32}   # used by the CPU pass

echo "==================== environment ===================="
echo "host: $(hostname)  date: $(date)"
module list 2>&1
echo "which julia: $(which julia)"
julia --version
for v in JULIA_DEPOT_PATH JULIA_NUM_THREADS ONEAPI_LTS JULIA_LOAD_PATH ZE_FLAT_DEVICE_HIERARCHY \
         ZE_AFFINITY_MASK NEO_CACHE_DIR NEO_CACHE_PERSISTENT IGC_DisableSOAPromotion \
         LD_LIBRARY_PATH; do
  printf '%s=%s\n' "$v" "${!v}"
done
echo "====================================================="
