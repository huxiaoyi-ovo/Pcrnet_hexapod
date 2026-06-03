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
START_JOY=1
START_CAMERA=1
START_PCR=1
START_ROSCORE=1
START_MUX=0
USE_TMUX=1
FULL_MONITOR=0
PCR_USR_COMMAND_TOPIC="/usr/command_pcr"
MANUAL_USR_COMMAND_TOPIC="/usr/command_manual"
MUX_OUTPUT_TOPIC="/usr/command"
MUX_RATE_HZ="50"
MUX_PCR_TIMEOUT_S="0.35"
MUX_MANUAL_TIMEOUT_S="0.2"
CAMERA_HEIGHT_M="0.37"
CAMERA_PITCH_DOWN_DEG="15"

CAMERA_EXTRA_ARGS=()
PCR_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash interface/scripts/pcr_real/run_pcr_real_compat.sh [options]

This starts the PCR real-robot compatibility path:
  D435i -> /pcr/target_state + /pcr/local_map_2ch
  joy_ctrl -> /usr/command_manual
  PCR      -> /usr/command_pcr
  run_agent2.py selects manual or PCR -> /sita_des

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
  --publish_cmd                Actually publish /usr/command_pcr from PCR.
  --no_run_agent               Do not start run_agent2.py.
  --no_joy                     Do not start joy_node and joy_ctrl.
  --no_camera                  Do not start D435i observation publisher.
  --no_pcr                     Do not start PCR node.
  --no_roscore                 Do not auto-start roscore if missing.
  --no_tmux                    Run in the old single-shell background mode.
  --full_monitor               In tmux mode, show each node/topic in its own pane.
  --no_mux                     Kept for old command lines; PCR now always publishes to /usr/command_pcr.
  --pcr_usr_command_topic TOP  Default: /usr/command_pcr
  --manual_usr_command_topic T Default: /usr/command_manual
  --mux_output_topic TOP       Deprecated.
  --mux_rate_hz HZ             Deprecated.
  --mux_pcr_timeout_s SEC      Deprecated.
  --mux_manual_timeout_s SEC   Deprecated.
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
        --no_joy) START_JOY=0; shift ;;
        --no_camera) START_CAMERA=0; shift ;;
        --no_pcr) START_PCR=0; shift ;;
        --no_roscore) START_ROSCORE=0; shift ;;
        --no_tmux) USE_TMUX=0; shift ;;
        --full_monitor) FULL_MONITOR=1; shift ;;
        --no_mux) START_MUX=0; shift ;;
        --pcr_usr_command_topic) PCR_USR_COMMAND_TOPIC="$2"; shift 2 ;;
        --manual_usr_command_topic) MANUAL_USR_COMMAND_TOPIC="$2"; shift 2 ;;
        --mux_output_topic) MUX_OUTPUT_TOPIC="$2"; shift 2 ;;
        --mux_rate_hz) MUX_RATE_HZ="$2"; shift 2 ;;
        --mux_pcr_timeout_s) MUX_PCR_TIMEOUT_S="$2"; shift 2 ;;
        --mux_manual_timeout_s) MUX_MANUAL_TIMEOUT_S="$2"; shift 2 ;;
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
if [[ "${START_JOY}" -eq 1 ]]; then
    command -v rosrun >/dev/null || { echo "rosrun not found; source the src_real catkin workspace first." >&2; exit 2; }
    rospack find interface >/dev/null || { echo "ROS package 'interface' not found; source the src_real catkin workspace first." >&2; exit 2; }
    rospack find joy >/dev/null || { echo "ROS package 'joy' not found; install ros-noetic-joy or pass --no_joy." >&2; exit 2; }
fi

START_MUX=0

if [[ "${START_RUN_AGENT}" -eq 1 ]] && command -v rosnode >/dev/null && timeout 1s rosnode list 2>/dev/null | grep -qE '(^|/)run_agent2(_[0-9]+_[0-9]+)?$'; then
    echo "Refuse to start: run_agent2 is already running." >&2
    echo "If manage.launch already started run_agent2 for hand initialization, rerun this script with --no_run_agent." >&2
    exit 2
fi
if [[ "${START_JOY}" -eq 1 ]] && command -v rosnode >/dev/null && timeout 1s rosnode list 2>/dev/null | grep -qE '(^|/)(joy_ctrl|joy_node)$'; then
    echo "Refuse to start: joy_ctrl or joy_node is already running." >&2
    echo "Stop the old hand-control launch first, or rerun this script with --no_joy." >&2
    exit 2
fi

quote_cmd() {
    printf "%q " "$@"
}

run_prefix() {
    local q_workspace
    q_workspace="$(printf "%q" "${WORKSPACE_DIR}")"
    cat <<EOF
cd ${q_workspace}
source /opt/ros/noetic/setup.zsh 2>/dev/null || source /opt/ros/noetic/setup.bash
source devel/setup.zsh 2>/dev/null || source devel/setup.bash 2>/dev/null || true
EOF
}

write_pane_script() {
    local path="$1"
    local title="$2"
    local body="$3"
    {
        echo "#!/usr/bin/env zsh"
        echo "setopt NO_NOMATCH"
        run_prefix
        echo "PIDS=()"
        echo "echo '[PCRRealCompat] ${title}'"
        echo "${body}"
        echo "echo"
        echo "echo '[PCRRealCompat] ${title} exited. Press Ctrl-D to close this pane.'"
        echo "exec zsh"
    } > "${path}"
    chmod +x "${path}"
}

write_tmux_wrapper_script() {
    local path="$1"
    local title="$2"
    shift 2
    {
        echo "#!/usr/bin/env zsh"
        echo "setopt NO_NOMATCH"
        run_prefix
        echo "PIDS=()"
        echo "echo '[PCRRealCompat] ${title}'"
        for item in "$@"; do
            local name="${item%%:::*}"
            local cmd="${item#*:::}"
            echo "echo '[PCRRealCompat] starting ${name}'"
            echo "${cmd} &"
            echo "PIDS+=(\$!)"
            echo "sleep 1"
        done
        echo "trap 'for pid in \${PIDS[@]}; do kill \$pid 2>/dev/null || true; done; exit 0' INT TERM EXIT"
        echo "wait"
        echo "echo"
        echo "echo '[PCRRealCompat] ${title} exited. Press Ctrl-D to close this pane.'"
        echo "exec zsh"
    } > "${path}"
    chmod +x "${path}"
}

add_tmux_pane() {
    local script_path="$1"
    local q_script_path
    q_script_path="$(printf "%q" "${script_path}")"
    if [[ "${TMUX_PANE_COUNT}" -eq 0 ]]; then
        tmux new-session -d -s "${TMUX_SESSION}" -n pcr "zsh -lc ${q_script_path}"
    else
        tmux split-window -t "${TMUX_SESSION}:0" "zsh -lc ${q_script_path}"
        tmux select-layout -t "${TMUX_SESSION}:0" tiled >/dev/null
    fi
    TMUX_PANE_COUNT=$((TMUX_PANE_COUNT + 1))
}

if [[ "${USE_TMUX}" -eq 1 && -z "${PCR_REAL_COMPAT_IN_TMUX:-}" ]]; then
    command -v tmux >/dev/null || {
        echo "tmux not found. Install it once with: sudo apt install -y tmux" >&2
        echo "Or rerun this script with --no_tmux." >&2
        exit 2
    }

    TMUX_SESSION="pcr_real_compat"
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        TMUX_SESSION="pcr_real_compat_$(date +%H%M%S)"
    fi
    TMUX_DIR="$(mktemp -d /tmp/pcr_real_compat_tmux.XXXXXX)"
    TMUX_PANE_COUNT=0
    WAIT_ROS="until timeout 1s rostopic list >/dev/null 2>&1; do echo '[PCRRealCompat] waiting for roscore...'; sleep 1; done"

    if ! timeout 1s rostopic list >/dev/null 2>&1; then
        if [[ "${START_ROSCORE}" -eq 1 ]]; then
            write_pane_script "${TMUX_DIR}/00_roscore.zsh" "roscore" "roscore"
            add_tmux_pane "${TMUX_DIR}/00_roscore.zsh"
        else
            echo "roscore is not running." >&2
            exit 2
        fi
    fi

    NODE_BG_ITEMS=()
    if [[ "${START_RUN_AGENT}" -eq 1 ]]; then
        RUN_AGENT_CMD=(rosrun interface run_agent2.py --agent="${LOWLEVEL_AGENT}" --device="${LOWLEVEL_DEVICE}" --manual_command_topic="${MANUAL_USR_COMMAND_TOPIC}" --pcr_command_topic="${PCR_USR_COMMAND_TOPIC}")
        RUN_AGENT_BODY="${WAIT_ROS}; $(quote_cmd "${RUN_AGENT_CMD[@]}")"
        if [[ "${FULL_MONITOR}" -eq 1 || "${FULL_MONITOR}" -eq 0 ]]; then
            write_pane_script "${TMUX_DIR}/10_run_agent2.zsh" "run_agent2" "${RUN_AGENT_BODY}"
            add_tmux_pane "${TMUX_DIR}/10_run_agent2.zsh"
        fi
    fi
    if [[ "${START_JOY}" -eq 1 ]]; then
        JOY_NODE_CMD=(rosrun joy joy_node)
        JOY_CTRL_CMD=(rosrun interface joy_ctrl _command_topic:="${MANUAL_USR_COMMAND_TOPIC}")
        JOY_NODE_BODY="${WAIT_ROS}; $(quote_cmd "${JOY_NODE_CMD[@]}")"
        JOY_CTRL_BODY="${WAIT_ROS}; $(quote_cmd "${JOY_CTRL_CMD[@]}")"
        if [[ "${FULL_MONITOR}" -eq 1 ]]; then
            write_pane_script "${TMUX_DIR}/20_joy_node.zsh" "joy_node" "${JOY_NODE_BODY}"
            write_pane_script "${TMUX_DIR}/21_joy_ctrl.zsh" "joy_ctrl" "${JOY_CTRL_BODY}"
            add_tmux_pane "${TMUX_DIR}/20_joy_node.zsh"
            add_tmux_pane "${TMUX_DIR}/21_joy_ctrl.zsh"
        else
            NODE_BG_ITEMS+=("joy_node:::${JOY_NODE_BODY}")
            NODE_BG_ITEMS+=("joy_ctrl:::${JOY_CTRL_BODY}")
        fi
    fi
    if [[ "${START_CAMERA}" -eq 1 ]]; then
        CAMERA_CMD=(python3 "${CODE_DIR}/real_pcr_input_check.py" --yolo_model "${YOLO_MODEL}" --width 640 --height 480 --fps 30 --map_size 32 --map_extent_m 3.0 --camera_height_m "${CAMERA_HEIGHT_M}" --camera_pitch_down_deg "${CAMERA_PITCH_DOWN_DEG}" --yolo_conf 0.35 --target_depth_mode roi --ground_remove_height_m 0.04 --debug_map_px 260)
        if [[ "${FILE_BRIDGE}" -eq 1 ]]; then
            CAMERA_CMD+=(--obs_file "${OBS_FILE}")
        else
            CAMERA_CMD+=(--publish_ros)
        fi
        if [[ "${SHOW_CAMERA}" -eq 1 ]]; then
            CAMERA_CMD+=(--show)
        fi
        CAMERA_CMD+=("${CAMERA_EXTRA_ARGS[@]}")
        CAMERA_BODY="${WAIT_ROS}; $(quote_cmd "${CAMERA_CMD[@]}")"
        if [[ "${FULL_MONITOR}" -eq 1 ]]; then
            write_pane_script "${TMUX_DIR}/30_camera.zsh" "D435i PCR observation" "${CAMERA_BODY}"
            add_tmux_pane "${TMUX_DIR}/30_camera.zsh"
        else
            NODE_BG_ITEMS+=("camera:::${CAMERA_BODY}")
        fi
    fi
    if [[ "${START_PCR}" -eq 1 ]]; then
        PCR_CMD=(python3 "${CODE_DIR}/pcr_realplay.py" --pcr_ckpt "${PCR_CKPT}" --avoid_ckpt "${AVOID_CKPT}" --lowlevel_ckpt "${LOWLEVEL_CKPT}" --cmd_backend usr_command --usr_command_topic "${PCR_USR_COMMAND_TOPIC}" --device "${PCR_DEVICE}" --rate_hz "${RATE_HZ}" --risk_memory --max_cmd_x 0.06 --max_cmd_y 0.10 --max_cmd_yaw 0.20)
        if [[ "${FILE_BRIDGE}" -eq 1 ]]; then
            PCR_CMD+=(--obs_file "${OBS_FILE}")
        fi
        if [[ "${PUBLISH_CMD}" -eq 1 ]]; then
            PCR_CMD+=(--publish_cmd)
        fi
        PCR_CMD+=("${PCR_EXTRA_ARGS[@]}")
        PCR_BODY="${WAIT_ROS}; $(quote_cmd "${PCR_CMD[@]}")"
        if [[ "${FULL_MONITOR}" -eq 1 ]]; then
            write_pane_script "${TMUX_DIR}/40_pcr_realplay.zsh" "PCR realplay" "${PCR_BODY}"
            add_tmux_pane "${TMUX_DIR}/40_pcr_realplay.zsh"
        else
            NODE_BG_ITEMS+=("pcr_realplay:::${PCR_BODY}")
        fi
    fi

    Q_MANUAL_TOPIC="$(printf "%q" "${MANUAL_USR_COMMAND_TOPIC}")"
    Q_PCR_TOPIC="$(printf "%q" "${PCR_USR_COMMAND_TOPIC}")"
    if [[ "${FULL_MONITOR}" -eq 0 ]]; then
        if [[ "${#NODE_BG_ITEMS[@]}" -gt 0 ]]; then
            write_tmux_wrapper_script "${TMUX_DIR}/20_support_nodes.zsh" "support nodes: joy camera PCR" "${NODE_BG_ITEMS[@]}"
            add_tmux_pane "${TMUX_DIR}/20_support_nodes.zsh"
        fi
    fi

    COMMAND_MONITOR_BODY="${WAIT_ROS}; while true; do clear; echo '--- /usr/command_manual'; timeout 1s rostopic echo -n 1 ${Q_MANUAL_TOPIC} || true; echo '--- /usr/command_pcr'; timeout 1s rostopic echo -n 1 ${Q_PCR_TOPIC} || true; sleep 0.2; done"
    ROBOT_MONITOR_BODY="${WAIT_ROS}; while true; do clear; echo '--- /sita_des'; timeout 1s rostopic echo -n 1 /sita_des || true; echo '--- /pcr/target_state'; timeout 1s rostopic echo -n 1 /pcr/target_state || true; echo '--- nodes/topics'; rosnode list 2>/dev/null | grep -E 'run_agent2|joy|pcr|camera|rosout' || true; rostopic list 2>/dev/null | grep -E 'usr|pcr|sita' || true; sleep 0.5; done"
    write_pane_script "${TMUX_DIR}/50_command_monitor.zsh" "command monitor" "${COMMAND_MONITOR_BODY}"
    write_pane_script "${TMUX_DIR}/51_robot_monitor.zsh" "robot state monitor" "${ROBOT_MONITOR_BODY}"
    add_tmux_pane "${TMUX_DIR}/50_command_monitor.zsh"
    add_tmux_pane "${TMUX_DIR}/51_robot_monitor.zsh"
    if [[ "${FULL_MONITOR}" -eq 1 ]]; then
        write_pane_script "${TMUX_DIR}/52_pcr_state_monitor.zsh" "PCR target monitor" "${WAIT_ROS}; rostopic echo /pcr/target_state"
        write_pane_script "${TMUX_DIR}/53_ros_monitor.zsh" "ROS node/topic monitor" "${WAIT_ROS}; watch -n 0.5 \"echo nodes; rosnode list; echo; echo topics; rostopic list | grep -E 'usr|pcr|sita'\""
        add_tmux_pane "${TMUX_DIR}/52_pcr_state_monitor.zsh"
        add_tmux_pane "${TMUX_DIR}/53_ros_monitor.zsh"
    fi
    tmux select-layout -t "${TMUX_SESSION}:0" tiled >/dev/null
    echo "[PCRRealCompat] attached tmux session: ${TMUX_SESSION}"
    echo "[PCRRealCompat] detach: Ctrl-b then d; stop all panes: tmux kill-session -t ${TMUX_SESSION}"
    tmux attach-session -t "${TMUX_SESSION}"
    exit 0
fi

NEEDS_ROS=0
if [[ "${FILE_BRIDGE}" -eq 0 && ( "${START_CAMERA}" -eq 1 || "${START_PCR}" -eq 1 || "${START_RUN_AGENT}" -eq 1 || "${START_JOY}" -eq 1 ) ]]; then
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
        --device="${LOWLEVEL_DEVICE}" \
        --manual_command_topic="${MANUAL_USR_COMMAND_TOPIC}" \
        --pcr_command_topic="${PCR_USR_COMMAND_TOPIC}"
fi

if [[ "${START_JOY}" -eq 1 ]]; then
    start_bg "joy_node" \
        rosrun joy joy_node
    start_bg "joy_ctrl" \
        rosrun interface joy_ctrl \
        _command_topic:="${MANUAL_USR_COMMAND_TOPIC}"
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
        --usr_command_topic "${PCR_USR_COMMAND_TOPIC}"
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
