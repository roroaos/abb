from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

# ============================================================
# ABB 통신 설정
# ============================================================
# ABB 로봇은 TCP 서버, 이 파이썬 프로그램은 TCP 클라이언트로 동작한다.
ROBOT_IP = os.getenv("ABB_ROBOT_IP", "192.168.3.2")
ROBOT_PORT = int(os.getenv("ABB_ROBOT_PORT", "1025"))

# 휴대폰 앱 또는 브라우저가 접속할 Flask 서버 설정이다.
HTTP_HOST = "0.0.0.0"
HTTP_PORT = int(os.getenv("ABB_HTTP_PORT", "5000"))
PC_IP = os.getenv("ABB_PC_IP", "192.168.3.71")

# RAPID가 줄바꿈 없는 문자열을 사용하므로 기본값은 False로 둔다.
SEND_NEWLINE = os.getenv("ABB_SEND_NEWLINE", "0") == "1"

# ============================================================
# 카메라 및 판별 설정
# ============================================================
CAMERA_INDEX = int(os.getenv("ABB_CAMERA_INDEX", "1"))
CAM_WIDTH = int(os.getenv("ABB_CAMERA_WIDTH", "1280"))
CAM_HEIGHT = int(os.getenv("ABB_CAMERA_HEIGHT", "720"))
JPEG_QUALITY = int(os.getenv("ABB_JPEG_QUALITY", "80"))

# 로컬 PC에도 OpenCV 창을 표시할지 결정한다.
SHOW_LOCAL_WINDOW = os.getenv("ABB_SHOW_LOCAL_WINDOW", "1") == "1"

# REQ 수신 후 새 판별에 사용할 시간과 안정화 조건이다.
REQUEST_MAX_WAIT = float(os.getenv("ABB_REQUEST_MAX_WAIT", "3.0"))
REQUEST_DISCARD_TIME = float(os.getenv("ABB_REQUEST_DISCARD_TIME", "0.25"))
MOTION_THRESHOLD = float(os.getenv("ABB_MOTION_THRESHOLD", "6.0"))
MOTION_STABLE_FRAMES = int(os.getenv("ABB_MOTION_STABLE_FRAMES", "3"))
RESULT_BUFFER_SIZE = int(os.getenv("ABB_RESULT_BUFFER_SIZE", "5"))
RESULT_REQUIRED_COUNT = int(os.getenv("ABB_RESULT_REQUIRED_COUNT", "4"))
PREVIEW_BUFFER_SIZE = 7

# 설정 파일은 기본적으로 파이썬 파일과 같은 폴더에 저장한다.
DEFAULT_CONFIG_FILE = Path(__file__).resolve().with_name("color_config.json")
CONFIG_FILE = Path(os.getenv("ABB_COLOR_CONFIG", str(DEFAULT_CONFIG_FILE)))

# PowerShell 화면을 보기 쉽게 Flask 접속 로그를 숨긴다.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)


@dataclass
class BridgeState:
    # ABB 통신 상태
    connected: bool = False
    phase: str = "starting"
    last_robot_message: str = "-"
    last_reply: str = "-"
    sock: Optional[socket.socket] = None

    # 설정에 저장된 세트 수와 현재 회차에 고정된 세트 수
    saved_set_count: Optional[int] = None
    active_set_count: int = 0
    batch_number: int = 0

    # 현재 분류 회차 진행 상태
    target_panels: int = 0
    processed_requests: int = 0
    normal_count: int = 0
    white_count: int = 0
    gray_count: int = 0
    yellow_count: int = 0
    failed_count: int = 0

    # 카메라 및 미리보기 상태
    camera_connected: bool = False
    roi_ok: bool = False
    sample_ok: bool = False
    preview_raw_label: str = "UNKNOWN"
    preview_stable_label: str = "UNKNOWN"
    h: float = 0.0
    s: float = 0.0
    v: float = 0.0
    motion_score: float = 0.0
    scene_stable: bool = False

    # ABB에 마지막으로 전달한 실제 판별 결과
    last_result_label: str = "UNKNOWN"
    last_result_code: str = "4"

    last_update: str = "-"
    events: Deque[str] = field(default_factory=lambda: deque(maxlen=200))
    lock: threading.RLock = field(default_factory=threading.RLock)


state = BridgeState()

# 설정 데이터는 카메라 스레드와 Flask 요청이 함께 사용한다.
config_lock = threading.RLock()
config_data: dict = {}

# 최신 카메라 프레임은 판별 작업과 영상 스트리밍이 함께 사용한다.
frame_condition = threading.Condition(threading.RLock())
latest_raw_frame: Optional[np.ndarray] = None
latest_jpeg: Optional[bytes] = None
latest_frame_seq: int = 0
latest_frame_time: float = 0.0


# ============================================================
# 공통 유틸리티
# ============================================================
def now() -> str:
    return time.strftime("%H:%M:%S")


def push_event(message: str) -> None:
    line = f"[{now()}] {message}"
    with state.lock:
        state.events.append(line)
        state.last_update = now()
    print(line, flush=True)


def plain(text: str, status: int = 200) -> Response:
    return Response(text, status=status, content_type="text/plain; charset=utf-8")


def phase_to_korean(phase: str) -> str:
    mapping = {
        "starting": "시작 중",
        "connecting": "ABB 연결 시도 중",
        "connected_waiting": "ABB 요청 대기",
        "sorting": "벽면 분류 진행 중",
        "assembly_wait": "분류 완료·조립 중 대기",
        "setup_required": "세트 수 설정 필요",
        "camera_setup_required": "카메라 설정 필요",
        "classifying": "새 프레임 판별 중",
        "disconnected": "ABB 통신 끊김",
    }
    return mapping.get(phase, phase)


def label_to_korean(label: str) -> str:
    mapping = {
        "WHITE": "흰색",
        "GRAY": "회색",
        "YELLOW": "노란색",
        "UNKNOWN": "판별 실패",
    }
    return mapping.get(label, label)


def label_to_rapid_code(label: str) -> str:
    if label == "WHITE":
        return "1"
    if label == "GRAY":
        return "2"
    if label == "YELLOW":
        return "3"
    return "4"


def get_latest_frame_copy() -> tuple[Optional[np.ndarray], int]:
    with frame_condition:
        frame = None if latest_raw_frame is None else latest_raw_frame.copy()
        return frame, latest_frame_seq


# ============================================================
# 설정 파일 관리
# ============================================================
def default_config() -> dict:
    return {
        "roi": None,
        "samples": {},
        "set_count": 2,
    }


def load_config() -> dict:
    data = default_config()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception as exc:
            print(f"설정 파일 읽기 실패: {exc}", flush=True)

    if not isinstance(data.get("samples"), dict):
        data["samples"] = {}

    saved_count = data.get("set_count")
    if not isinstance(saved_count, int) or not 1 <= saved_count <= 3:
        data["set_count"] = None

    return data


def save_config_locked() -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(config_data, file, ensure_ascii=False, indent=4)

    temporary.replace(CONFIG_FILE)


def refresh_config_state() -> None:
    with config_lock:
        roi = config_data.get("roi")
        samples = config_data.get("samples", {})
        saved_count = config_data.get("set_count")

    with state.lock:
        state.saved_set_count = saved_count
        state.roi_ok = isinstance(roi, dict)
        state.sample_ok = all(color in samples for color in ("white", "gray", "yellow"))


def set_saved_count(value: int) -> tuple[bool, str]:
    if not 1 <= value <= 3:
        return False, "세트 수는 1~3만 저장할 수 있습니다."

    with config_lock:
        config_data["set_count"] = value
        save_config_locked()

    refresh_config_state()
    push_event(f"앱 설정에 세트 수 저장: {value}")
    return True, f"{value}세트 저장 완료"


def set_roi(x: int, y: int, width: int, height: int) -> tuple[bool, str]:
    frame, _ = get_latest_frame_copy()
    if frame is None:
        return False, "카메라 프레임이 없습니다."

    frame_height, frame_width = frame.shape[:2]
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(0, min(width, frame_width - x))
    height = max(0, min(height, frame_height - y))

    if width < 10 or height < 10:
        return False, "ROI 영역이 너무 작습니다."

    with config_lock:
        config_data["roi"] = {"x": x, "y": y, "w": width, "h": height}
        save_config_locked()

    refresh_config_state()
    push_event(f"ROI 저장: x={x}, y={y}, w={width}, h={height}")
    return True, "ROI 저장 완료"


def clear_roi() -> None:
    with config_lock:
        config_data["roi"] = None
        save_config_locked()

    refresh_config_state()
    push_event("ROI 삭제")


def save_color_sample(color_name: str) -> tuple[bool, str]:
    if color_name not in ("white", "gray", "yellow"):
        return False, "지원 색상은 white, gray, yellow입니다."

    frame, _ = get_latest_frame_copy()
    if frame is None:
        return False, "카메라 프레임이 없습니다."

    with config_lock:
        roi = config_data.get("roi")

    if not isinstance(roi, dict):
        return False, "ROI를 먼저 설정하세요."

    roi_image = crop_roi(frame, roi)
    if roi_image is None:
        return False, "ROI 영역을 읽을 수 없습니다."

    feature = extract_hsv_feature(roi_image)

    with config_lock:
        samples = config_data.setdefault("samples", {})
        samples[color_name] = feature
        save_config_locked()

    refresh_config_state()
    push_event(f"{label_to_korean(color_name.upper())} 기준값 저장")
    return True, f"{label_to_korean(color_name.upper())} 기준값 저장 완료"


def clear_samples() -> None:
    with config_lock:
        config_data["samples"] = {}
        save_config_locked()

    refresh_config_state()
    push_event("흰색·회색·노란색 기준값 전체 삭제")


# ============================================================
# OpenCV 색상 판별
# ============================================================
def crop_roi(frame: np.ndarray, roi: dict) -> Optional[np.ndarray]:
    try:
        x = int(roi["x"])
        y = int(roi["y"])
        width = int(roi["w"])
        height = int(roi["h"])
    except (KeyError, TypeError, ValueError):
        return None

    frame_height, frame_width = frame.shape[:2]
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(0, min(width, frame_width - x))
    height = max(0, min(height, frame_height - y))

    if width <= 0 or height <= 0:
        return None

    image = frame[y:y + height, x:x + width]
    if image.size == 0:
        return None

    return image


def extract_hsv_feature(roi_image: np.ndarray) -> dict:
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # 지나치게 어두운 픽셀은 판별에서 제외한다.
    pixels = pixels[pixels[:, 2] > 30]

    if len(pixels) == 0:
        return {
            "h": 0.0,
            "s": 0.0,
            "v": 0.0,
            "mean_h": 0.0,
            "mean_s": 0.0,
            "mean_v": 0.0,
        }

    return {
        "h": float(np.median(pixels[:, 0])),
        "s": float(np.median(pixels[:, 1])),
        "v": float(np.median(pixels[:, 2])),
        "mean_h": float(np.mean(pixels[:, 0])),
        "mean_s": float(np.mean(pixels[:, 1])),
        "mean_v": float(np.mean(pixels[:, 2])),
    }


def hue_distance(first: float, second: float) -> float:
    difference = abs(first - second)
    return min(difference, 180 - difference)


def color_distance(current: dict, sample: dict, color_name: str) -> float:
    h = current["h"]
    s = current["s"]
    v = current["v"]
    sample_h = sample["h"]
    sample_s = sample["s"]
    sample_v = sample["v"]

    # 흰색과 회색은 채도와 밝기 차이를 중심으로 비교한다.
    if color_name in ("white", "gray"):
        return abs(s - sample_s) * 1.0 + abs(v - sample_v) * 2.5

    # 노란색은 색상값 차이도 크게 반영한다.
    if color_name == "yellow":
        return (
            hue_distance(h, sample_h) * 2.0
            + abs(s - sample_s) * 0.8
            + abs(v - sample_v) * 0.5
        )

    return 9999.0


def classify_by_saved_samples(feature: dict, samples: dict) -> str:
    if not all(color in samples for color in ("white", "gray", "yellow")):
        return "UNKNOWN"

    scores = {
        color_name: color_distance(feature, sample, color_name)
        for color_name, sample in samples.items()
        if color_name in ("white", "gray", "yellow")
    }

    if not scores:
        return "UNKNOWN"

    best_color = min(scores, key=scores.get)
    mapping = {
        "white": "WHITE",
        "gray": "GRAY",
        "yellow": "YELLOW",
    }
    return mapping.get(best_color, "UNKNOWN")


def get_stable_result(buffer: Deque[str], required_count: Optional[int] = None) -> str:
    if not buffer:
        return "UNKNOWN"

    label, count = Counter(buffer).most_common(1)[0]
    minimum = required_count if required_count is not None else len(buffer) // 2 + 1

    if label != "UNKNOWN" and count >= minimum:
        return label

    return "UNKNOWN"


def calculate_motion_score(previous_gray: np.ndarray, current_gray: np.ndarray) -> float:
    difference = cv2.absdiff(previous_gray, current_gray)
    return float(np.mean(difference))


def classify_for_rapid_request() -> str:
    # REQ마다 이전 판별 결과를 사용하지 않고 새 프레임으로 독립 판별한다.
    with config_lock:
        roi = config_data.get("roi")
        samples = dict(config_data.get("samples", {}))

    with state.lock:
        camera_ready = state.camera_connected
        roi_ready = state.roi_ok
        samples_ready = state.sample_ok
        state.phase = "classifying"
        state.scene_stable = False

    if not camera_ready or not roi_ready or not samples_ready:
        with state.lock:
            state.phase = "camera_setup_required"
        push_event("REQ 판별 실패: 카메라·ROI·색상 기준값 설정 필요")
        return "UNKNOWN"

    deadline = time.monotonic() + REQUEST_MAX_WAIT
    discard_until = time.monotonic() + REQUEST_DISCARD_TIME
    previous_gray: Optional[np.ndarray] = None
    stable_motion_frames = 0
    result_buffer: Deque[str] = deque(maxlen=RESULT_BUFFER_SIZE)

    with frame_condition:
        seen_seq = latest_frame_seq

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()

        with frame_condition:
            frame_condition.wait_for(
                lambda: latest_frame_seq > seen_seq,
                timeout=max(0.0, remaining),
            )

            if latest_frame_seq <= seen_seq or latest_raw_frame is None:
                continue

            frame = latest_raw_frame.copy()
            seen_seq = latest_frame_seq

        roi_image = crop_roi(frame, roi)
        if roi_image is None:
            break

        # REQ 직후 로봇이나 물체가 아직 움직일 수 있으므로 초기 프레임은 버린다.
        if time.monotonic() < discard_until:
            continue

        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if previous_gray is None:
            previous_gray = gray
            continue

        motion_score = calculate_motion_score(previous_gray, gray)
        previous_gray = gray

        with state.lock:
            state.motion_score = motion_score

        if motion_score <= MOTION_THRESHOLD:
            stable_motion_frames += 1
        else:
            # 로봇이나 물체가 ROI를 가리고 지나가면 기존 후보를 모두 폐기한다.
            stable_motion_frames = 0
            result_buffer.clear()
            with state.lock:
                state.scene_stable = False
            continue

        if stable_motion_frames < MOTION_STABLE_FRAMES:
            continue

        with state.lock:
            state.scene_stable = True

        feature = extract_hsv_feature(roi_image)
        label = classify_by_saved_samples(feature, samples)
        result_buffer.append(label)

        with state.lock:
            state.h = feature["h"]
            state.s = feature["s"]
            state.v = feature["v"]

        if len(result_buffer) >= RESULT_BUFFER_SIZE:
            stable_label = get_stable_result(result_buffer, RESULT_REQUIRED_COUNT)
            if stable_label != "UNKNOWN":
                return stable_label

    push_event("REQ 판별 실패: 3초 안에 안정된 색상 결과를 얻지 못함")
    return "UNKNOWN"


# ============================================================
# ABB 소켓 통신
# ============================================================
def close_robot_socket() -> None:
    with state.lock:
        sock = state.sock
        state.sock = None
        state.connected = False
        state.phase = "disconnected"

    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def receive_text(sock: socket.socket) -> str:
    # 조립과 재투입 시간이 길어도 연결을 끊지 않도록 수신 시간 제한을 두지 않는다.
    sock.settimeout(None)
    data = sock.recv(1024)

    if not data:
        raise ConnectionError("RAPID가 연결을 종료했습니다.")

    text = data.decode("ascii", errors="replace").strip("\x00\r\n ")

    with state.lock:
        state.last_robot_message = text

    push_event(f"RAPID -> PC: {text}")
    return text


def send_text(sock: socket.socket, text: str) -> None:
    try:
        payload = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("RAPID 전송은 영문·숫자·기호만 사용할 수 있습니다.") from exc

    if SEND_NEWLINE:
        payload += b"\n"

    sock.sendall(payload)

    with state.lock:
        state.last_reply = text

    push_event(f"PC -> RAPID: {text}")


def start_new_batch() -> int:
    with state.lock:
        saved_count = state.saved_set_count

        state.processed_requests = 0
        state.normal_count = 0
        state.white_count = 0
        state.gray_count = 0
        state.yellow_count = 0
        state.failed_count = 0
        state.last_result_label = "UNKNOWN"
        state.last_result_code = "4"

        if saved_count is None:
            state.active_set_count = 0
            state.target_panels = 0
            state.phase = "setup_required"
            return 0

        # 현재 회차에 사용할 수량을 고정하여 조립 중 앱 변경이 다음 회차부터 반영되게 한다.
        state.active_set_count = saved_count
        state.target_panels = saved_count * 4
        state.batch_number += 1
        state.phase = "sorting"
        return saved_count


def record_request_result(label: str) -> None:
    code = label_to_rapid_code(label)

    with state.lock:
        state.last_result_label = label
        state.last_result_code = code
        state.processed_requests += 1

        if label == "WHITE":
            state.white_count += 1
            state.normal_count += 1
        elif label == "GRAY":
            state.gray_count += 1
            state.normal_count += 1
        elif label == "YELLOW":
            state.yellow_count += 1
            state.normal_count += 1
        else:
            state.failed_count += 1

        if state.target_panels > 0 and state.processed_requests >= state.target_panels:
            state.phase = "assembly_wait"
        else:
            state.phase = "sorting"


def auto_reply_for(received: str) -> str:
    command = received.strip().upper()

    if command == "HELLO":
        with state.lock:
            state.phase = "connected_waiting"
        return "OK"

    if command == "HOW_MANY":
        set_count = start_new_batch()
        if set_count == 0:
            push_event("저장된 세트 수 없음: HOW_MANY에 0 응답")
        else:
            push_event(
                f"새 분류 회차 시작: {set_count}세트, 벽면 {set_count * 4}개"
            )
        return str(set_count)

    if command == "REQ":
        with state.lock:
            valid_batch = state.active_set_count > 0 and state.target_panels > 0

        if not valid_batch:
            push_event("REQ 판별 실패: 먼저 HOW_MANY 회차 설정 필요")
            label = "UNKNOWN"
        else:
            label = classify_for_rapid_request()

        record_request_result(label)
        code = label_to_rapid_code(label)
        push_event(f"판별 결과: {label_to_korean(label)} -> {code}")
        return code

    if command == "STATUS":
        with state.lock:
            return (
                f"PHASE={state.phase},SET={state.active_set_count},"
                f"DONE={state.processed_requests},TARGET={state.target_panels},"
                f"LAST={state.last_result_code}"
            )

    return "NG"


def handle_robot_session(sock: socket.socket) -> None:
    push_event("자동 응답 시작: HELLO=OK, HOW_MANY=저장 세트 수, REQ=새 색상 판별")

    while True:
        received = receive_text(sock)
        reply = auto_reply_for(received)
        send_text(sock, reply)


def robot_worker() -> None:
    push_event(f"ABB 연결 대기: {ROBOT_IP}:{ROBOT_PORT}")

    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.settimeout(3.0)

        try:
            with state.lock:
                state.phase = "connecting"

            sock.connect((ROBOT_IP, ROBOT_PORT))
            sock.settimeout(None)

            with state.lock:
                state.sock = sock
                state.connected = True
                state.phase = "connected_waiting"

            push_event(f"ABB 연결 성공: {ROBOT_IP}:{ROBOT_PORT}")
            handle_robot_session(sock)

        except (OSError, ConnectionError, ValueError) as exc:
            with state.lock:
                was_connected = state.connected

            if was_connected:
                push_event(f"ABB 연결 종료: {exc}")

            time.sleep(2.0)

        finally:
            close_robot_socket()


# ============================================================
# 카메라 작업
# ============================================================
def draw_camera_overlay(frame: np.ndarray, roi: Optional[dict]) -> np.ndarray:
    display = frame.copy()

    with state.lock:
        preview_raw = state.preview_raw_label
        preview_stable = state.preview_stable_label
        last_result = state.last_result_label
        phase_text = phase_to_korean(state.phase)
        h_value = state.h
        s_value = state.s
        v_value = state.v
        motion = state.motion_score

    if isinstance(roi, dict):
        try:
            x = int(roi["x"])
            y = int(roi["y"])
            width = int(roi["w"])
            height = int(roi["h"])
            cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
        except (KeyError, TypeError, ValueError):
            pass

    lines = [
        f"PREVIEW RAW: {preview_raw}  STABLE: {preview_stable}",
        f"LAST RAPID RESULT: {last_result} / CODE {label_to_rapid_code(last_result)}",
        f"HSV H={h_value:.1f} S={s_value:.1f} V={v_value:.1f}",
        f"MOTION={motion:.2f}  PHASE={phase_text}",
    ]

    for index, line in enumerate(lines):
        y_position = 35 + index * 35
        cv2.putText(
            display,
            line,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
        )

    return display


def handle_local_key(key: int, frame: np.ndarray, window_name: str) -> bool:
    if key == ord("s"):
        selected = cv2.selectROI(window_name, frame, False, False)
        x, y, width, height = selected
        if width > 0 and height > 0:
            success, message = set_roi(int(x), int(y), int(width), int(height))
            print(message, flush=True)
            if not success:
                push_event(message)

    elif key == ord("w"):
        print(save_color_sample("white")[1], flush=True)

    elif key == ord("g"):
        print(save_color_sample("gray")[1], flush=True)

    elif key == ord("y"):
        print(save_color_sample("yellow")[1], flush=True)

    elif key == ord("d"):
        clear_samples()

    elif key == ord("c"):
        print(json.dumps(status_payload(), ensure_ascii=False, indent=2), flush=True)

    elif key == ord("q"):
        return False

    return True


def camera_worker() -> None:
    global latest_raw_frame, latest_jpeg, latest_frame_seq, latest_frame_time

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        with state.lock:
            state.camera_connected = False
        push_event("USB 카메라 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    with state.lock:
        state.camera_connected = True

    window_name = "ABB Wall Color Bridge"
    if SHOW_LOCAL_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("--------------------------------", flush=True)
    print("로컬 카메라 키", flush=True)
    print("s : ROI 선택 및 저장", flush=True)
    print("w : 흰색 기준값 저장", flush=True)
    print("g : 회색 기준값 저장", flush=True)
    print("y : 노란색 기준값 저장", flush=True)
    print("c : 현재 상태 출력", flush=True)
    print("d : 기준값 전체 삭제", flush=True)
    print("q : 로컬 카메라 창 종료", flush=True)
    print("--------------------------------", flush=True)

    preview_buffer: Deque[str] = deque(maxlen=PREVIEW_BUFFER_SIZE)
    keep_running = True

    while keep_running:
        success, frame = cap.read()
        if not success:
            push_event("카메라 프레임 읽기 실패")
            break

        with config_lock:
            roi = config_data.get("roi")
            samples = dict(config_data.get("samples", {}))

        raw_label = "UNKNOWN"
        stable_label = "UNKNOWN"

        if isinstance(roi, dict):
            roi_image = crop_roi(frame, roi)
            if roi_image is not None:
                feature = extract_hsv_feature(roi_image)
                raw_label = classify_by_saved_samples(feature, samples)
                preview_buffer.append(raw_label)
                stable_label = get_stable_result(preview_buffer)

                with state.lock:
                    state.h = feature["h"]
                    state.s = feature["s"]
                    state.v = feature["v"]
        else:
            preview_buffer.clear()

        with state.lock:
            state.preview_raw_label = raw_label
            state.preview_stable_label = stable_label
            state.camera_connected = True

        display = draw_camera_overlay(frame, roi)
        encode_success, encoded = cv2.imencode(
            ".jpg",
            display,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

        if encode_success:
            with frame_condition:
                latest_raw_frame = frame.copy()
                latest_jpeg = encoded.tobytes()
                latest_frame_seq += 1
                latest_frame_time = time.time()
                frame_condition.notify_all()

        if SHOW_LOCAL_WINDOW:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            keep_running = handle_local_key(key, frame, window_name)

    cap.release()

    if SHOW_LOCAL_WINDOW:
        cv2.destroyAllWindows()

    with state.lock:
        state.camera_connected = False

    push_event("카메라 작업 종료")


# ============================================================
# 앱 및 브라우저용 상태 데이터
# ============================================================
def status_payload() -> dict:
    with config_lock:
        roi = config_data.get("roi")
        samples = config_data.get("samples", {})

    with state.lock:
        return {
            "abb_connected": state.connected,
            "phase": state.phase,
            "phase_text": phase_to_korean(state.phase),
            "saved_set_count": state.saved_set_count,
            "active_set_count": state.active_set_count,
            "batch_number": state.batch_number,
            "target_panels": state.target_panels,
            "processed_requests": state.processed_requests,
            "normal_count": state.normal_count,
            "white_count": state.white_count,
            "gray_count": state.gray_count,
            "yellow_count": state.yellow_count,
            "failed_count": state.failed_count,
            "camera_connected": state.camera_connected,
            "roi_ok": state.roi_ok,
            "sample_ok": state.sample_ok,
            "roi": roi,
            "samples": {
                "white": "white" in samples,
                "gray": "gray" in samples,
                "yellow": "yellow" in samples,
            },
            "preview_raw_label": state.preview_raw_label,
            "preview_stable_label": state.preview_stable_label,
            "last_result_label": state.last_result_label,
            "last_result_text": label_to_korean(state.last_result_label),
            "last_result_code": state.last_result_code,
            "h": round(state.h, 1),
            "s": round(state.s, 1),
            "v": round(state.v, 1),
            "motion_score": round(state.motion_score, 2),
            "scene_stable": state.scene_stable,
            "last_robot_message": state.last_robot_message,
            "last_reply": state.last_reply,
            "last_update": state.last_update,
            "events": list(state.events)[-30:],
        }


# ============================================================
# 앱과 브라우저용 HTTP 기능
# ============================================================
@app.get("/")
@app.get("/camera")
def camera_page() -> Response:
    html = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ABB 벽면 색상 판별</title>
<style>
body { margin:0; font-family:Arial,sans-serif; background:#111; color:#eee; }
main { max-width:980px; margin:auto; padding:14px; }
h1 { font-size:21px; margin:4px 0 12px; }
.camera-wrap { position:relative; width:100%; background:#000; }
#video { display:block; width:100%; height:auto; }
#overlay { position:absolute; left:0; top:0; width:100%; height:100%; touch-action:none; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:10px; }
button,input { min-height:46px; font-size:16px; border-radius:8px; border:1px solid #555; }
button { background:#2d2d2d; color:#fff; }
button:active { background:#555; }
input { width:100%; box-sizing:border-box; padding:8px; background:#fff; color:#111; }
.card { margin-top:10px; background:#1d1d1d; padding:12px; border-radius:9px; white-space:pre-wrap; line-height:1.5; }
.full { grid-column:1 / -1; }
.small { font-size:13px; color:#bbb; }
</style>
</head>
<body>
<main>
<h1>ABB 벽면 색상 판별</h1>
<div class="camera-wrap">
  <img id="video" src="/video_feed" alt="카메라 영상">
  <canvas id="overlay"></canvas>
</div>
<div class="small">영상 위를 드래그하면 ROI가 저장됩니다.</div>
<div class="grid">
  <button onclick="saveSample('white')">흰색 기준 저장</button>
  <button onclick="saveSample('gray')">회색 기준 저장</button>
  <button onclick="saveSample('yellow')">노란색 기준 저장</button>
  <button onclick="resetSamples()">기준값 전체 삭제</button>
  <button onclick="clearRoi()">ROI 삭제</button>
  <button onclick="refreshStatus()">상태 새로고침</button>
  <input id="setCount" type="number" min="1" max="3" placeholder="세트 수 1~3">
  <button onclick="saveSetCount()">세트 수 저장</button>
</div>
<div id="message" class="card">준비 중...</div>
<div id="status" class="card">상태 불러오는 중...</div>
</main>
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
let dragging = false;
let startX = 0, startY = 0;

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
}
video.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);
setInterval(resizeCanvas, 1500);

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(rect.width, event.clientX - rect.left)),
    y: Math.max(0, Math.min(rect.height, event.clientY - rect.top))
  };
}

canvas.addEventListener('pointerdown', event => {
  dragging = true;
  canvas.setPointerCapture(event.pointerId);
  const point = pointFromEvent(event);
  startX = point.x;
  startY = point.y;
});

canvas.addEventListener('pointermove', event => {
  if (!dragging) return;
  const point = pointFromEvent(event);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 3;
  ctx.strokeRect(startX, startY, point.x - startX, point.y - startY);
});

canvas.addEventListener('pointerup', async event => {
  if (!dragging) return;
  dragging = false;
  const point = pointFromEvent(event);
  const x1 = Math.min(startX, point.x);
  const y1 = Math.min(startY, point.y);
  const x2 = Math.max(startX, point.x);
  const y2 = Math.max(startY, point.y);
  const scaleX = (video.naturalWidth || 1280) / canvas.width;
  const scaleY = (video.naturalHeight || 720) / canvas.height;
  const payload = {
    x: Math.round(x1 * scaleX),
    y: Math.round(y1 * scaleY),
    w: Math.round((x2 - x1) * scaleX),
    h: Math.round((y2 - y1) * scaleY)
  };
  const response = await fetch('/api/roi', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
  });
  showMessage(await response.text());
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  refreshStatus();
});

function showMessage(text) {
  document.getElementById('message').textContent = text;
}

async function saveSample(color) {
  const response = await fetch('/api/sample/' + color, {method:'POST'});
  showMessage(await response.text());
  refreshStatus();
}

async function resetSamples() {
  const response = await fetch('/api/samples/reset', {method:'POST'});
  showMessage(await response.text());
  refreshStatus();
}

async function clearRoi() {
  const response = await fetch('/api/roi/clear', {method:'POST'});
  showMessage(await response.text());
  refreshStatus();
}

async function saveSetCount() {
  const value = Number(document.getElementById('setCount').value);
  const response = await fetch('/api/set_count', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({value:value})
  });
  showMessage(await response.text());
  refreshStatus();
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    if (data.saved_set_count) document.getElementById('setCount').value = data.saved_set_count;
    document.getElementById('status').textContent =
`ABB 연결: ${data.abb_connected ? '연결됨' : '끊김'}
현재 상태: ${data.phase_text}
저장 세트 수: ${data.saved_set_count ?? '미설정'}
현재 회차 세트 수: ${data.active_set_count}
분류 진행: ${data.processed_requests} / ${data.target_panels}
흰색: ${data.white_count}  회색: ${data.gray_count}  노란색: ${data.yellow_count}
판별 실패: ${data.failed_count}
최근 판별: ${data.last_result_text} / 코드 ${data.last_result_code}
카메라: ${data.camera_connected ? '정상' : '오류'}
ROI: ${data.roi_ok ? '설정됨' : '미설정'}
기준값: ${data.sample_ok ? '모두 설정됨' : '미완료'}
HSV: H${data.h} S${data.s} V${data.v}
움직임 점수: ${data.motion_score}
마지막 수신: ${data.last_robot_message}
마지막 송신: ${data.last_reply}`;
  } catch (error) {
    document.getElementById('status').textContent = '상태 연결 실패: ' + error;
  }
}

refreshStatus();
setInterval(refreshStatus, 1000);
</script>
</body>
</html>
"""
    return Response(html, content_type="text/html; charset=utf-8")


@app.get("/video_feed")
def video_feed() -> Response:
    def generate():
        seen_seq = -1
        while True:
            with frame_condition:
                frame_condition.wait_for(
                    lambda: latest_frame_seq != seen_seq and latest_jpeg is not None,
                    timeout=5.0,
                )

                if latest_jpeg is None:
                    continue

                jpeg = latest_jpeg
                seen_seq = latest_frame_seq

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot.jpg")
def snapshot() -> Response:
    with frame_condition:
        if latest_jpeg is None:
            return plain("카메라 프레임이 없습니다.", 503)
        jpeg = latest_jpeg
    return Response(jpeg, content_type="image/jpeg")


@app.get("/api/status")
def api_status() -> Response:
    return jsonify(status_payload())


@app.get("/status")
def status_text() -> Response:
    data = status_payload()
    text = (
        f"ABB연결={data['abb_connected']}\n"
        f"상태={data['phase_text']}\n"
        f"저장세트수={data['saved_set_count'] if data['saved_set_count'] is not None else '미설정'}\n"
        f"현재세트수={data['active_set_count']}\n"
        f"진행={data['processed_requests']}/{data['target_panels']}\n"
        f"흰색={data['white_count']}\n"
        f"회색={data['gray_count']}\n"
        f"노란색={data['yellow_count']}\n"
        f"판별실패={data['failed_count']}\n"
        f"최근판별={data['last_result_text']}\n"
        f"RAPID코드={data['last_result_code']}\n"
        f"카메라={'OK' if data['camera_connected'] else '오류'}\n"
        f"ROI={'OK' if data['roi_ok'] else '미설정'}\n"
        f"기준값={'OK' if data['sample_ok'] else '미설정'}\n"
        f"HSV=H{data['h']},S{data['s']},V{data['v']}\n"
        f"마지막수신={data['last_robot_message']}\n"
        f"마지막송신={data['last_reply']}\n"
        f"시간={data['last_update']}"
    )
    return plain(text)


@app.post("/api/set_count")
def api_set_count() -> Response:
    payload = request.get_json(silent=True) or {}
    try:
        value = int(payload.get("value"))
    except (TypeError, ValueError):
        return plain("세트 수는 1~3의 숫자로 입력하세요.", 400)

    success, message = set_saved_count(value)
    return plain(message, 200 if success else 400)


@app.get("/setcount/<int:value>")
def set_count_compatibility(value: int) -> Response:
    success, message = set_saved_count(value)
    return plain(message, 200 if success else 400)


@app.post("/api/roi")
def api_roi() -> Response:
    payload = request.get_json(silent=True) or {}

    try:
        x = int(payload.get("x"))
        y = int(payload.get("y"))
        width = int(payload.get("w"))
        height = int(payload.get("h"))
    except (TypeError, ValueError):
        return plain("ROI 좌표가 올바르지 않습니다.", 400)

    success, message = set_roi(x, y, width, height)
    return plain(message, 200 if success else 400)


@app.post("/api/roi/clear")
def api_clear_roi() -> Response:
    clear_roi()
    return plain("ROI 삭제 완료")


@app.route("/api/sample/<color_name>", methods=["GET", "POST"])
def api_save_sample(color_name: str) -> Response:
    success, message = save_color_sample(color_name.lower().strip())
    return plain(message, 200 if success else 400)


@app.post("/api/samples/reset")
def api_reset_samples() -> Response:
    clear_samples()
    return plain("색상 기준값 전체 삭제 완료")


@app.get("/rapid_event")
def rapid_event() -> Response:
    with state.lock:
        message = state.events.popleft() if state.events else "NONE"
    return plain(message)


@app.get("/start")
def start_status() -> Response:
    return plain("자동 응답 실행 중: HELLO=OK, HOW_MANY=저장 세트 수, REQ=새 색상 판별")


@app.get("/reset")
def reset_progress() -> Response:
    with state.lock:
        state.events.clear()
        state.processed_requests = 0
        state.normal_count = 0
        state.white_count = 0
        state.gray_count = 0
        state.yellow_count = 0
        state.failed_count = 0
        state.last_result_label = "UNKNOWN"
        state.last_result_code = "4"
        state.phase = "connected_waiting" if state.connected else "disconnected"

    push_event("앱에서 진행 기록 초기화")
    return plain("진행 기록 초기화 완료")


@app.get("/test/<color_name>")
def test_color(color_name: str) -> Response:
    mapping = {
        "white": "WHITE",
        "gray": "GRAY",
        "grey": "GRAY",
        "yellow": "YELLOW",
        "unknown": "UNKNOWN",
    }
    label = mapping.get(color_name.lower().strip())
    if label is None:
        return plain("지원 색상: white / gray / yellow / unknown", 400)

    record_request_result(label)
    code = label_to_rapid_code(label)
    push_event(f"테스트 판별 설정: {label_to_korean(label)} -> {code}")
    return plain(f"OK: {label_to_korean(label)}, code={code}")


# ============================================================
# 프로그램 시작
# ============================================================
if __name__ == "__main__":
    with config_lock:
        config_data = load_config()

    refresh_config_state()

    print("================================", flush=True)
    print("ABB 벽면 색상 판별 브리지", flush=True)
    print(f"앱/브라우저 : http://{PC_IP}:{HTTP_PORT}/", flush=True)
    print(f"상태 확인   : http://{PC_IP}:{HTTP_PORT}/status", flush=True)
    print(f"ABB 대상    : {ROBOT_IP}:{ROBOT_PORT}", flush=True)
    print("응답 규칙   : HELLO→OK, HOW_MANY→저장 세트 수, REQ→1/2/3/4", flush=True)
    print("색상 코드   : 흰색=1, 회색=2, 노란색=3, 판별 실패=4", flush=True)
    print("================================", flush=True)

    threading.Thread(
        target=camera_worker,
        name="카메라 작업",
        daemon=True,
    ).start()

    threading.Thread(
        target=robot_worker,
        name="ABB 통신 작업",
        daemon=True,
    ).start()

    app.run(
        host=HTTP_HOST,
        port=HTTP_PORT,
        threaded=True,
        use_reloader=False,
    )
