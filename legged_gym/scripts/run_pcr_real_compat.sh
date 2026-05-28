#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../../interface" && -d "${SCRIPT_DIR}/../../../agent" ]]; then
    WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    CODE_DIR="${SCRIPT_DIR}"
    ASSET_DIR="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/../../src_real/interface/scripts/pcr_real" ]]; then
    WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    CODE_DIR="${WORKSPACE_DIR}/src_real/interface/scripts/pcr_real"
    ASSET_DIR="${CODE_DIR}"
else
    WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    CODE_DIR="${SCRIPT_DIR}"
    ASSET_DIR="${SCRIPT_DIR}"
fi
cd "${WORKSPACE_DIR}"

PCR_CKPT="${ASSET_DIR}/checkpoints/moe_teacher_best_learnedw.pt"
AVOID_CKPT="${ASSET_DIR}/checkpoints/avoid_best.pt"
LOWLEVEL_CKPT="${ASSET_DIR}/checkpoints/low_level_best.pt"
YOLO_MODEL="${ASSET_DIR}/yolov8n.pt"
LOWLEVEL_AGENT="base_line200.pt"
LOWLEVEL_DEVICE="cpu"
PCR_DEVICE="cpu"
RATE_HZ="10"
FILE_BRIDGE=0
OBS_FILE="${WORKSPACE_DIR}/outputs/real_d435i_check/latest_obs.json"
SHOW_CAMERA=0
PUBLISH_CMD=0
START_RUN_AGENT=1
START_CAMERA=1
START_PCR=1
START_ROSCORE=1
CAMERA_HEIGHT_M="0.37"
CAMERA_PITCH_DOWN_DEG="15"

CAMERA_EXTRA_ARGS=()
PCR_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash interface/scripts/pcr_real/run_pcr_real_compat.sh [options]

This starts the PCR real-robot compatibility path without starting joy_ctrl:
  D435i -> /pcr/target_state + /pcr/local_map_2ch
  PCR   -> /usr/command
  run_agent2.py -> /sita_des

Safe default:
  PCR runs in dry-run mode. Add --publish_cmd only after checking outputs.

Options:
  --pcr_ckpt PATH              Default: interface/scripts/pcr_real/checkpoints/moe_teacher_best_learnedw.pt
  --avoid_ckpt PATH            Default: interface/scripts/pcr_real/checkpoints/avoid_best.pt
  --lowlevel_ckpt PATH         Default: interface/scripts/pcr_real/checkpoints/low_level_best.pt
  --yolo_model PATH            Default: interface/scripts/pcr_real/yolov8n.pt
  --lowlevel_agent NAME        Agent under /home/nvidia/agents for run_agent2.py.
  --lowlevel_device DEVICE     Default: cpu
  --pcr_device DEVICE          Default: cpu
  --rate_hz HZ                 Default: 10
  --file_bridge                Use JSON observation file instead of ROS topics.
  --obs_file PATH              Default: outputs/real_d435i_check/latest_obs.json
  --show                       Show D435i debug window.
  --publish_cmd                Actually publish /usr/command from PCR.
  --no_run_agent               Do not start run_agent2.py.
  --no_camera                  Do not start D435i observation publisher.
  --no_pcr                     Do not start PCR node.
  --no_roscore                 Do not auto-start roscore if missing.
  --camera_height_m VALUE      Default: 0.37
  --camera_pitch_down_deg VAL  Default: 15
  --camera_arg ARG             Append one raw argument to the camera command.
  --pcr_arg ARG                Append one raw argument to the PCR command.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pcr_ckpt) PCR_CKPT="$2"; shift 2 ;;
        --avoid_ckpt) AVOID_CKPT="$2"; shift 2 ;;
        --lowlevel_ckpt) LOWLEVEL_CKPT="$2"; shift 2 ;;
        --yolo_model) YOLO_MODEL="$2"; shift 2 ;;
        --lowlevel_agent) LOWLEVEL_AGENT="$2"; shift 2 ;;
        --lowlevel_device) LOWLEVEL_DEVICE="$2"; shift 2 ;;
        --pcr_device) PCR_DEVICE="$2"; shift 2 ;;
        --rate_hz) RATE_HZ="$2"; shift 2 ;;
        --file_bridge) FILE_BRIDGE=1; shift ;;
        --obs_file) OBS_FILE="$2"; shift 2 ;;
        --show) SHOW_CAMERA=1; shift ;;
        --publish_cmd) PUBLISH_CMD=1; shift ;;
        --no_run_agent) START_RUN_AGENT=0; shift ;;
        --no_camera) START_CAMERA=0; shift ;;
        --no_pcr) START_PCR=0; shift ;;
        --no_roscore) START_ROSCORE=0; shift ;;
        --camera_height_m) CAMERA_HEIGHT_M="$2"; shift 2 ;;
        --camera_pitch_down_deg) CAMERA_PITCH_DOWN_DEG="$2"; shift 2 ;;
        --camera_arg) CAMERA_EXTRA_ARGS+=("$2"); shift 2 ;;
        --pcr_arg) PCR_EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "${START_PCR}" -eq 1 && ! -f "${PCR_CKPT}" ]]; then
    echo "PCR checkpoint not found: ${PCR_CKPT}" >&2
    echo "Put it under ${ASSET_DIR}/checkpoints/ or pass --pcr_ckpt with the real path." >&2
    exit 2
fi
if [[ "${START_PCR}" -eq 1 && ! -f "${AVOID_CKPT}" ]]; then
    echo "Avoid checkpoint not found: ${AVOID_CKPT}" >&2
    exit 2
fi
if [[ "${START_PCR}" -eq 1 && ! -f "${LOWLEVEL_CKPT}" ]]; then
    echo "Low-level checkpoint hint not found: ${LOWLEVEL_CKPT}" >&2
    exit 2
fi
if [[ "${START_CAMERA}" -eq 1 && ! -f "${YOLO_MODEL}" ]]; then
    echo "YOLO model not found: ${YOLO_MODEL}" >&2
    exit 2
fi

if [[ "${START_RUN_AGENT}" -eq 1 ]]; then
    command -v rosrun >/dev/null || { echo "rosrun not found; source the src_real catkin workspace first." >&2; exit 2; }
    rospack find interface >/dev/null || { echo "ROS package 'interface' not found; source the src_real catkin workspace first." >&2; exit 2; }
    if [[ ! -f "/home/nvidia/agents/${LOWLEVEL_AGENT}" ]]; then
        echo "Low-level agent not found: /home/nvidia/agents/${LOWLEVEL_AGENT}" >&2
        echo "Install it once with:" >&2
        echo "  mkdir -p /home/nvidia/agents && cp ${WORKSPACE_DIR}/agent/${LOWLEVEL_AGENT} /home/nvidia/agents/${LOWLEVEL_AGENT}" >&2
        exit 2
    fi
fi

NEEDS_ROS=0
if [[ "${FILE_BRIDGE}" -eq 0 && ( "${START_CAMERA}" -eq 1 || "${START_PCR}" -eq 1 || "${START_RUN_AGENT}" -eq 1 ) ]]; then
    NEEDS_ROS=1
fi
if [[ "${PUBLISH_CMD}" -eq 1 || "${START_RUN_AGENT}" -eq 1 ]]; then
    NEEDS_ROS=1
fi

if [[ "${NEEDS_ROS}" -eq 1 ]]; then
    command -v rostopic >/dev/null || { echo "rostopic not found; source ROS or use --file_bridge for laptop checks." >&2; exit 2; }
    if ! timeout 1s rostopic list >/dev/null 2>&1; then
        if [[ "${START_ROSCORE}" -eq 1 ]]; then
            command -v roscore >/dev/null || { echo "roscore not found; source ROS or use --file_bridge for laptop checks." >&2; exit 2; }
            echo "[PCRRealCompat] roscore not detected; starting roscore..."
            roscore &
            ROSCORE_PID=$!
            sleep 2
        else
            echo "roscore is not running." >&2
            exit 2
        fi
    fi
fi

if command -v rosnode >/dev/null && rosnode list 2>/dev/null | grep -qE '(^|/)joy_ctrl$'; then
    echo "Refuse to start: joy_ctrl is already running and may publish /usr/command." >&2
    echo "Stop joy_ctrl first, or use manual mode instead of PCR mode." >&2
    exit 2
fi

PIDS=()
cleanup() {
    set +e
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${ROSCORE_PID:-}" ]]; then
        kill "${ROSCORE_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

start_bg() {
    local name="$1"
    shift
    echo "[PCRRealCompat] starting ${name}: $*"
    "$@" &
    PIDS+=("$!")
    sleep 1
}

if [[ "${START_RUN_AGENT}" -eq 1 ]]; then
    start_bg "run_agent2" \
        rosrun interface run_agent2.py \
        --agent="${LOWLEVEL_AGENT}" \
        --device="${LOWLEVEL_DEVICE}"
fi

if [[ "${START_CAMERA}" -eq 1 ]]; then
    CAMERA_CMD=(
        python3 "${CODE_DIR}/real_pcr_input_check.py"
        --yolo_model "${YOLO_MODEL}"
        --width 640
        --height 480
        --fps 30
        --map_size 32
        --map_extent_m 3.0
        --camera_height_m "${CAMERA_HEIGHT_M}"
        --camera_pitch_down_deg "${CAMERA_PITCH_DOWN_DEG}"
        --yolo_conf 0.35
        --target_depth_mode roi
        --ground_remove_height_m 0.04
        --debug_map_px 260
    )
    if [[ "${FILE_BRIDGE}" -eq 1 ]]; then
        CAMERA_CMD+=(--obs_file "${OBS_FILE}")
    else
        CAMERA_CMD+=(--publish_ros)
    fi
    if [[ "${SHOW_CAMERA}" -eq 1 ]]; then
        CAMERA_CMD+=(--show)
    fi
    CAMERA_CMD+=("${CAMERA_EXTRA_ARGS[@]}")
    start_bg "D435i PCR observation" "${CAMERA_CMD[@]}"
fi

if [[ "${START_PCR}" -eq 1 ]]; then
    PCR_CMD=(
        python3 "${CODE_DIR}/pcr_realplay.py"
        --pcr_ckpt "${PCR_CKPT}"
        --avoid_ckpt "${AVOID_CKPT}"
        --lowlevel_ckpt "${LOWLEVEL_CKPT}"
        --cmd_backend usr_command
        --device "${PCR_DEVICE}"
        --rate_hz "${RATE_HZ}"
        --risk_memory
        --max_cmd_x 0.06
        --max_cmd_y 0.10
        --max_cmd_yaw 0.20
    )
    if [[ "${FILE_BRIDGE}" -eq 1 ]]; then
        PCR_CMD+=(--obs_file "${OBS_FILE}")
    fi
    if [[ "${PUBLISH_CMD}" -eq 1 ]]; then
        PCR_CMD+=(--publish_cmd)
    fi
    PCR_CMD+=("${PCR_EXTRA_ARGS[@]}")
    start_bg "PCR realplay" "${PCR_CMD[@]}"
fi

echo "[PCRRealCompat] running. Press Ctrl+C to stop all started processes."
wait
