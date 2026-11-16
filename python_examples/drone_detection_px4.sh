#!/usr/bin/env bash
# Private PX4 resources: never install demo airframes into the user's build.
set -euo pipefail
if [[ ${1:-} == --stop ]]; then
    if [[ -f $2 ]]; then
        read -r pid < "$2"
        [[ $pid =~ ^[0-9]+$ ]] && kill -TERM "$pid" 2>/dev/null || true
    fi
    exit 0
fi
build=$1
firmware=$2
runtime=$3
instance=$4
sim_host=$5
api_host=$6
api_port=$7
[[ $build == '~/'* ]] && build="$HOME/${build:2}"
build=$(realpath "$build")
[[ -x $build/bin/px4 && -d $build/etc ]] || { echo 'Build PX4 SITL first'; exit 1; }
mkdir -p "$runtime"
runtime=$(realpath "$runtime")
cp -RL "$build/etc" "$runtime/etc"
airframe="$runtime/etc/init.d-posix/airframes/31000_pterosim_detection"
tr -d '\r' < "$firmware" > "$airframe"
cat >> "$airframe" <<'EOF'

# This demo has no RC input; retain PX4's estimator and arming checks.
param set-default COM_RC_IN_MODE 4
param set-default COM_DISARM_PRFLT 30
EOF
# An explicit API link also works across WSL NAT. Do not depend on PX4's
# default localhost-only Offboard destination or its port clamp after instance 8.
sed -i "/^mavlink start .*udp_offboard_port_local/c\\mavlink start -x -u $((14600 + instance)) -o $api_port -t $api_host -r 1000000 -m onboard" \
    "$runtime/etc/init.d-posix/px4-rc.mavlink"
cd "$runtime"
echo "$$" > px4.pid
export PX4_SYS_AUTOSTART=31000 PX4_SIM_MODEL=none_iris
export PX4_SIM_HOSTNAME="$sim_host" PX4_LOCKSTEP=1 PX4_SIM_SPEED_FACTOR=1
exec "$build/bin/px4" -i "$instance" -d "$runtime/etc"
