import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time
import math
import torch

# ==========================================
# 1. 坐标滤波器：One Euro Filter
# ==========================================
class OneEuroFilter:
    def __init__(self, te=0.033, min_cutoff=0.6, beta=0.01):
        self.x_prev, self.dx_prev = None, None
        self.te, self.min_cutoff, self.beta = te, min_cutoff, beta

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, 0.0
            return x
        alpha_d = 1.0 / (1.0 + 1.0 / (2 * math.pi * 1.0 * self.te))
        dx = (x - self.x_prev) / self.te
        edx = self.dx_prev + (alpha_d * (dx - self.dx_prev))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * self.te))
        x_filtered = self.x_prev + (alpha * (x - self.x_prev))
        self.x_prev, self.dx_prev = x_filtered, edx
        return x_filtered

# ==========================================
# 2. 语义解析
# ==========================================
def get_semantics(rx, ry, vx, vy):
    p_dist = "近" if rx < 1.2 else ("远" if rx > 2.2 else "中")
    p_side = "左" if ry > 0.4 else ("右" if ry < -0.4 else "中")
    v_th = 0.15 
    v_x_str = "out" if vx > v_th else ("in" if vx < -v_th else "")
    v_y_str = "left" if vy > v_th else ("right" if vy < -v_th else "")
    
    if not v_x_str and not v_y_str: move_label = "STAY"
    elif v_x_str and v_y_str: move_label = f"{v_y_str}_{v_x_str}"
    else: move_label = v_x_str + v_y_str
    return f"{p_side}{p_dist}", move_label

# ==========================================
# 3. 主程序
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt').to(device)
    PITCH, OFFSET = math.radians(10.0), 0.3
    
    f_rx, f_ry = OneEuroFilter(beta=0.005), OneEuroFilter(beta=0.005)
    f_vx, f_vy = OneEuroFilter(min_cutoff=0.1, beta=0.02), OneEuroFilter(min_cutoff=0.1, beta=0.02)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # --- 硬件级空洞填补滤波器 ---
    hole_filling = rs.hole_filling_filter()

    last_pos, last_time = None, time.time()
    last_valid_box = None
    lost_counter = 0

    print("[INFO] V9 启动：已修复深度空洞导致的 UI 渲染截断问题。")

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f: continue

            # 应用硬件滤波填补黑洞
            depth_f = hole_filling.process(depth_f).as_depth_frame()
            
            img = np.asanyarray(color_f.get_data())
            depth_image = np.asanyarray(depth_f.get_data())

            results = model.predict(img, classes=0, conf=0.6, verbose=False, device=device, half=True)

            current_box = None
            if len(results) > 0 and len(results[0].boxes) > 0:
                current_box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                last_valid_box = current_box
                lost_counter = 0
            elif last_valid_box is not None and lost_counter < 5:
                current_box = last_valid_box
                lost_counter += 1
            else:
                last_valid_box = None

            if current_box is not None:
                x1, y1, x2, y2 = current_box
                u, v = (x1 + x2) // 2, y2 - 10
                
                # ==========================================
                # 核心修复 1：无论深度是否有效，YOLO框绝对不闪！
                # ==========================================
                box_color = (0, 255, 0) if lost_counter == 0 else (0, 200, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(img, (u, v), 12, (0, 0, 255), -1) 

                # ==========================================
                # 核心修复 2：区域中值采样对抗深度空洞
                # ==========================================
                # 取 10x10 的区域
                u_min, u_max = max(0, u-5), min(640, u+5)
                v_min, v_max = max(0, v-5), min(480, v+5)
                patch = depth_image[v_min:v_max, u_min:u_max]
                
                # 过滤掉 0 值和噪点 (转换为米)
                valid_depths = patch[(patch > 0)] * depth_scale
                
                if len(valid_depths) > 0:
                    dist = np.median(valid_depths) # 取中位数，最稳健
                    
                    if 0.1 < dist < 5.0:
                        now = time.time()
                        dt = now - last_time
                        pt_c = rs.rs2_deproject_pixel_to_point(intrin, [u, v], dist)
                        
                        rx_raw = pt_c[2]*math.cos(PITCH) + pt_c[1]*math.sin(PITCH) + OFFSET
                        ry_raw = -pt_c[0]
                        
                        srx, sry = f_rx.filter(rx_raw), f_ry.filter(ry_raw)
                        vx, vy = 0.0, 0.0
                        if last_pos is not None and dt > 0:
                            vx = f_vx.filter((srx - last_pos[0]) / dt)
                            vy = f_vy.filter((sry - last_pos[1]) / dt)
                        
                        last_pos, last_time = (srx, sry), now
                        pos_str, move_str = get_semantics(srx, sry, vx, vy)

                        # UI 绘制动作
                        cv2.putText(img, f"Action: {move_str}", (x1, y2 + 25), 1, 1.5, (255, 255, 0), 2)
                        cv2.putText(img, f"Pos: {pos_str}", (x1, y2 + 55), 1, 1.5, box_color, 2)
                        if move_str != "STAY":
                            cv2.arrowedLine(img, (u, v), (int(u-vy*80), int(v-vx*80)), (255, 0, 0), 3)
                else:
                    # 深度采空时的提示，但框依然在！
                    cv2.putText(img, "DEPTH HOLE", (x1, y2 + 25), 1, 1.5, (0, 0, 255), 2)

            cv2.rectangle(img, (0, 430), (640, 480), (0, 0, 0), -1)
            info = f"DIST: {last_pos[0]:.2f}m | SIDE: {last_pos[1]:.2f}m" if last_pos else "SEARCHING..."
            cv2.putText(img, info, (15, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow('Hexapod V9 - Architecture Fixed', img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()