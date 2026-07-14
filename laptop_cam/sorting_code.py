import cv2
import numpy as np
import json
import threading
import time
import socket

from pathlib import Path
from collections import deque, Counter
from datetime import datetime

from flask import Flask
from flask_cors import CORS


# ============================================================
# 기본 설정
# ============================================================
PC_IP = "192.168.3.71"

HTTP_PORT = 5000
RAPID_PORT = 1025

CAMERA_INDEX = 1
CAM_WIDTH = 1280
CAM_HEIGHT = 720

CONFIG_FILE = Path("color_config.json")
STABLE_BUFFER_SIZE = 7


# ============================================================
# Flask / 상태값
# ============================================================
app = Flask(__name__)
CORS(app)

lock = threading.Lock()

state = {
    "running": False,
    "emergency": False,
    "manual_mode": False,

    "last_cmd": "없음",
    "color": "unknown",

    "stable_label": "UNKNOWN",
    "raw_label": "UNKNOWN",

    "roi_ok": False,
    "sample_ok": False,

    "h": 0.0,
    "s": 0.0,
    "v": 0.0,

    "set_cnt": 0,

    "rapid_connected": False,
    "rapid_last_cmd": "없음",

    "last_update": ""
}


# ============================================================
# 공통 함수
# ============================================================
def now():
    return datetime.now().strftime("%H:%M:%S")


def label_to_send(label):
    if label == "WHITE":
        return "white"
    if label == "GRAY":
        return "gray"
    if label == "YELLOW":
        return "yellow"
    return "unknown"


def label_to_rapid_code(label):
    if label == "WHITE":
        return "1"
    if label == "GRAY":
        return "2"
    if label == "YELLOW":
        return "3"
    return "0"


# ============================================================
# 설정 파일
# ============================================================
def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"설정 파일 읽기 실패: {e}")

    return {
        "roi": None,
        "samples": {}
    }


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


# ============================================================
# 색상 판별 함수
# ============================================================
def extract_hsv_feature(roi_img):
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # 너무 어두운 픽셀 제외
    pixels = pixels[pixels[:, 2] > 30]

    if len(pixels) == 0:
        return {
            "h": 0.0,
            "s": 0.0,
            "v": 0.0,
            "mean_h": 0.0,
            "mean_s": 0.0,
            "mean_v": 0.0
        }

    return {
        "h": float(np.median(pixels[:, 0])),
        "s": float(np.median(pixels[:, 1])),
        "v": float(np.median(pixels[:, 2])),
        "mean_h": float(np.mean(pixels[:, 0])),
        "mean_s": float(np.mean(pixels[:, 1])),
        "mean_v": float(np.mean(pixels[:, 2]))
    }


def hue_distance(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def color_distance(current, sample, color_name):
    h = current["h"]
    s = current["s"]
    v = current["v"]

    sh = sample["h"]
    ss = sample["s"]
    sv = sample["v"]

    # 흰색/회색은 S, V 차이를 우선
    if color_name in ["white", "gray"]:
        return (
            abs(s - ss) * 1.0 +
            abs(v - sv) * 2.5
        )

    # 노랑은 H도 중요
    if color_name == "yellow":
        return (
            hue_distance(h, sh) * 2.0 +
            abs(s - ss) * 0.8 +
            abs(v - sv) * 0.5
        )

    return 9999


def classify_by_saved_samples(feature, samples):
    # 저장된 기준값이 있으면 기준값 거리 비교
    if len(samples) > 0:
        scores = {}

        for color_name, sample in samples.items():
            scores[color_name] = color_distance(feature, sample, color_name)

        best_color = min(scores, key=scores.get)

        if best_color == "white":
            return "WHITE", "white"

        if best_color == "gray":
            return "GRAY", "gray"

        if best_color == "yellow":
            return "YELLOW", "yellow"

    # 기준값 없을 때 임시 HSV 판별
    h = feature["h"]
    s = feature["s"]
    v = feature["v"]

    if 18 <= h <= 42 and s > 50 and v > 60:
        return "YELLOW", "yellow"

    if s < 80 and v > 175:
        return "WHITE", "white"

    if s < 90 and 60 <= v <= 175:
        return "GRAY", "gray"

    return "UNKNOWN", "unknown"


def get_stable_result(buffer):
    if not buffer:
        return "UNKNOWN"

    counter = Counter(buffer)
    label, count = counter.most_common(1)[0]

    if count >= len(buffer) // 2 + 1:
        return label

    return "UNKNOWN"


# ============================================================
# Flask HTTP API
# ============================================================
@app.route("/")
def home():
    return "ABB 카메라-앱-RAPID 연동 서버 실행중"


@app.route("/status")
def status():
    with lock:
        if state["emergency"]:
            mode = "비상정지"
        elif state["running"]:
            mode = "실행중"
        else:
            mode = "대기중"

        return (
            f"상태: {mode}\n"
            f"판별: {state['color']}\n"
            f"원본: {state['raw_label']}\n"
            f"안정값: {state['stable_label']}\n"
            f"HSV: H={state['h']:.1f}, S={state['s']:.1f}, V={state['v']:.1f}\n"
            f"ROI: {'OK' if state['roi_ok'] else '미설정'}\n"
            f"기준값: {'OK' if state['sample_ok'] else '미설정'}\n"
            f"세트 수: {state['set_cnt']}\n"
            f"수동모드: {'ON' if state['manual_mode'] else 'OFF'}\n"
            f"RAPID 연결: {'ON' if state['rapid_connected'] else 'OFF'}\n"
            f"RAPID 마지막 명령: {state['rapid_last_cmd']}\n"
            f"마지막 명령: {state['last_cmd']}\n"
            f"시간: {state['last_update']}"
        )


@app.route("/start")
def start():
    with lock:
        state["running"] = True
        state["emergency"] = False
        state["manual_mode"] = False
        state["last_cmd"] = "시작"
        state["last_update"] = now()
    return "OK: 시작"


@app.route("/stop")
def stop():
    with lock:
        state["running"] = False
        state["last_cmd"] = "정지"
        state["last_update"] = now()
    return "OK: 정지"


@app.route("/reset")
def reset():
    with lock:
        state["running"] = False
        state["emergency"] = False
        state["manual_mode"] = False

        state["color"] = "unknown"
        state["stable_label"] = "UNKNOWN"
        state["raw_label"] = "UNKNOWN"

        state["set_cnt"] = 0

        state["last_cmd"] = "초기화"
        state["last_update"] = now()

    return "OK: 초기화"


@app.route("/estop")
def estop():
    with lock:
        state["running"] = False
        state["emergency"] = True
        state["last_cmd"] = "비상정지"
        state["last_update"] = now()
    return "OK: 비상정지"


@app.route("/setcount/<int:n>")
def set_count(n):
    if n < 0:
        n = 0

    with lock:
        state["set_cnt"] = n
        state["last_cmd"] = f"{n}세트 설정"
        state["last_update"] = now()

    return f"OK: {n}세트"


# 앱 테스트 버튼용
@app.route("/test/white")
def test_white():
    with lock:
        state["manual_mode"] = True
        state["color"] = "white"
        state["stable_label"] = "WHITE"
        state["raw_label"] = "WHITE"
        state["last_cmd"] = "흰색 테스트"
        state["last_update"] = now()
    return "OK: 흰색"


@app.route("/test/gray")
def test_gray():
    with lock:
        state["manual_mode"] = True
        state["color"] = "gray"
        state["stable_label"] = "GRAY"
        state["raw_label"] = "GRAY"
        state["last_cmd"] = "회색 테스트"
        state["last_update"] = now()
    return "OK: 회색"


@app.route("/test/yellow")
def test_yellow():
    with lock:
        state["manual_mode"] = True
        state["color"] = "yellow"
        state["stable_label"] = "YELLOW"
        state["raw_label"] = "YELLOW"
        state["last_cmd"] = "노랑 테스트"
        state["last_update"] = now()
    return "OK: 노랑"


def run_flask_server():
    app.run(
        host="0.0.0.0",
        port=HTTP_PORT,
        debug=False,
        use_reloader=False
    )


# ============================================================
# RAPID TCP Socket 서버
# ============================================================
def rapid_socket_server():
    HOST = "0.0.0.0"
    PORT = RAPID_PORT

    def send_line(conn, text):
        conn.sendall((text + "\n").encode("utf-8"))

    while True:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            srv.bind((HOST, PORT))
            srv.listen(1)

            print(f"RAPID 소켓 서버 대기중: {HOST}:{PORT}")

            conn, addr = srv.accept()
            print(f"RAPID 연결됨: {addr}")

            with lock:
                state["rapid_connected"] = True
                state["rapid_last_cmd"] = "CONNECTED"
                state["last_update"] = now()

            try:
                while True:
                    data = conn.recv(1024)

                    if not data:
                        break

                    cmd = data.decode("utf-8", errors="ignore").strip().upper()

                    with lock:
                        state["rapid_last_cmd"] = cmd
                        state["last_update"] = now()

                    print(f"RAPID RX: {cmd}")

                    if cmd == "HELLO":
                        send_line(conn, "OK")

                    elif cmd == "HOW_MANY":
                        with lock:
                            cnt = state["set_cnt"]
                        send_line(conn, str(cnt))

                    elif cmd == "REQ":
                        with lock:
                            running = state["running"]
                            emergency = state["emergency"]
                            label = state["stable_label"]

                        # 실행중이 아니거나 비상정지면 0 응답
                        if not running or emergency:
                            send_line(conn, "0")
                        else:
                            send_line(conn, label_to_rapid_code(label))

                    elif cmd == "STATUS":
                        with lock:
                            color = state["color"]
                            stable = state["stable_label"]
                            cnt = state["set_cnt"]
                            running = state["running"]
                            emergency = state["emergency"]

                        send_line(
                            conn,
                            f"COLOR={color},STABLE={stable},SET={cnt},RUN={running},ESTOP={emergency}"
                        )

                    else:
                        send_line(conn, "NG")

            except Exception as e:
                print(f"RAPID 통신 오류: {e}")

            finally:
                conn.close()

                with lock:
                    state["rapid_connected"] = False
                    state["rapid_last_cmd"] = "DISCONNECTED"
                    state["last_update"] = now()

                print("RAPID 연결 종료. 재대기")

        except Exception as e:
            print(f"RAPID 소켓 서버 오류: {e}")
            time.sleep(1)

        finally:
            try:
                srv.close()
            except:
                pass


# ============================================================
# 카메라 루프
# ============================================================
def camera_loop():
    config = load_config()

    roi_rect = None
    if config.get("roi") is not None:
        r = config["roi"]
        roi_rect = (
            int(r["x"]),
            int(r["y"]),
            int(r["w"]),
            int(r["h"])
        )

    samples = config.get("samples", {})
    result_buffer = deque(maxlen=STABLE_BUFFER_SIZE)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("USB 카메라 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    window_name = "ABB Camera App Rapid Server"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("--------------------------------")
    print("카메라 서버 실행")
    print("s : ROI 선택/저장")
    print("w : 현재 ROI를 흰색 기준값으로 저장")
    print("g : 현재 ROI를 회색 기준값으로 저장")
    print("y : 현재 ROI를 노랑 기준값으로 저장")
    print("c : 현재 상태 출력")
    print("d : 기준값 삭제")
    print("q : 종료")
    print("--------------------------------")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        display = frame.copy()

        with lock:
            running = state["running"]
            emergency = state["emergency"]
            manual_mode = state["manual_mode"]

        sample_ok = all(k in samples for k in ["white", "gray", "yellow"])

        with lock:
            state["roi_ok"] = roi_rect is not None
            state["sample_ok"] = sample_ok

        if roi_rect is None:
            cv2.putText(
                display,
                "Press S to select ROI",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        else:
            x, y, w, h = roi_rect
            roi_img = frame[y:y + h, x:x + w]

            if roi_img.size > 0:
                feature = extract_hsv_feature(roi_img)
                raw_label, raw_send = classify_by_saved_samples(feature, samples)

                result_buffer.append(raw_label)
                stable_label = get_stable_result(result_buffer)
                stable_send = label_to_send(stable_label)

                # raw/stable/HSV는 항상 업데이트
                with lock:
                    state["stable_label"] = stable_label
                    state["raw_label"] = raw_label
                    state["h"] = feature["h"]
                    state["s"] = feature["s"]
                    state["v"] = feature["v"]
                    state["last_update"] = now()

                # 앱에 표시되는 최종 판별값은 실행중 + 비상정지 아님 + 수동모드 아님일 때만 자동 갱신
                if running and not emergency and not manual_mode:
                    with lock:
                        state["color"] = stable_send

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"RAW: {raw_label}  STABLE: {stable_label}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"SEND: {stable_send}  RAPID: {label_to_rapid_code(stable_label)}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"HSV H={feature['h']:.1f} S={feature['s']:.1f} V={feature['v']:.1f}",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"RUN={running} ESTOP={emergency} MANUAL={manual_mode}",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"Samples W={'white' in samples} G={'gray' in samples} Y={'yellow' in samples}",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            selected = cv2.selectROI(window_name, frame, False, False)
            x, y, w, h = selected

            if w > 0 and h > 0:
                roi_rect = (
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                )

                config["roi"] = {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                }

                save_config(config)
                result_buffer.clear()

                print(f"ROI 저장됨: x={x}, y={y}, w={w}, h={h}")

        elif key == ord("w"):
            if roi_rect is not None:
                x, y, w, h = roi_rect
                roi_img = frame[y:y + h, x:x + w]
                samples["white"] = extract_hsv_feature(roi_img)
                config["samples"] = samples
                save_config(config)
                result_buffer.clear()

                print("흰색 기준값 저장됨")
                print(samples["white"])

        elif key == ord("g"):
            if roi_rect is not None:
                x, y, w, h = roi_rect
                roi_img = frame[y:y + h, x:x + w]
                samples["gray"] = extract_hsv_feature(roi_img)
                config["samples"] = samples
                save_config(config)
                result_buffer.clear()

                print("회색 기준값 저장됨")
                print(samples["gray"])

        elif key == ord("y"):
            if roi_rect is not None:
                x, y, w, h = roi_rect
                roi_img = frame[y:y + h, x:x + w]
                samples["yellow"] = extract_hsv_feature(roi_img)
                config["samples"] = samples
                save_config(config)
                result_buffer.clear()

                print("노랑 기준값 저장됨")
                print(samples["yellow"])

        elif key == ord("c"):
            with lock:
                print("--------------------------------")
                print(f"state   : {state}")
                print(f"ROI     : {roi_rect}")
                print(f"samples : {samples}")
                print("--------------------------------")

        elif key == ord("d"):
            samples = {}
            config["samples"] = samples
            save_config(config)
            result_buffer.clear()

            print("기준값 삭제됨")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    print("ABB 카메라-앱-RAPID 통합 서버 시작")
    print(f"앱 주소        : http://{PC_IP}:{HTTP_PORT}/status")
    print(f"세트 수 설정   : http://{PC_IP}:{HTTP_PORT}/setcount/3")
    print(f"RAPID 접속 IP  : {PC_IP}")
    print(f"RAPID 접속 PORT: {RAPID_PORT}")

    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    socket_thread = threading.Thread(target=rapid_socket_server, daemon=True)
    socket_thread.start()

    time.sleep(1)

    camera_loop()
