import cv2
import time
import math as m
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import numpy as np
import psutil
from psutil import Process
import os

# ____CONFIG

CALIBRATION_DURATION = 5

THRESH_SHOULDER = 0.065
THRESH_NECK = 0.035
THRESH_BODY = 6

THRESH_NECK_ANGLE = 40
THRESH_TORSO_ANGLE = 10

SMOOTH_WINDOW = 5
DEADZONE = 0.01

# ____________HELPERS

def findAngle(x1, y1, x2, y2):
    try:
        return abs(m.degrees(m.atan2(x2 - x1, y1 - y2)))
    except:
        return 0

def mid(p1, p2):
    return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

def angle_2d(p1, p2):
    return m.degrees(m.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def shoulder_tilt(l, r):
    w = abs(r[0]-l[0])
    return (l[1]-r[1])/w if w else 0

def neck_offset(neck, l, r):
    mpt = mid(l,r)
    w = abs(r[0]-l[0])
    return (neck[0]-mpt[0])/w if w else 0

def body_axis(l, r, lh, rh):
    return angle_2d(mid(l,r), mid(lh,rh))
    
# _______________SIGNAL STABILIZATION

def smooth(value, history, window=SMOOTH_WINDOW):
    history.append(value)
    if len(history) > window:
        history.pop(0)
    return sum(history) / len(history)

def deadzone(v, dz=DEADZONE):
    return 0 if abs(v) < dz else v

# ____________UI HELPERS

def draw_panel(img, x, y, w, h, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w,y+h), (30,30,30), -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_progress(img, x, y, w, h, progress):
    cv2.rectangle(img, (x,y), (x+w,y+h), (80,80,80), 2)
    fill = int(w * progress)
    cv2.rectangle(img, (x,y), (x+fill,y+h), (0,255,255), -1)

def posture_color(is_bad):
    return (0,0,255) if is_bad else (0,255,0)

# ______________SKELETON
POSE_CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(12,14),(14,16)
]

def draw_skeleton(img, lm, w, h, color, mode="FRONT"):
    pts = {}
    for i,p in enumerate(lm):
        pts[i] = (int(p.x*w), int(p.y*h))
        cv2.circle(img, pts[i], 3, color, -1)

    connections = [(11,12),(11,23),(12,24),(23,24)] if mode=="FRONT" else POSE_CONNECTIONS
    for a,b in connections:
        cv2.line(img, pts[a], pts[b], color, 2)


# _________________GRAPH/DATA

t_series,v1,v2,v3 = [],[],[],[]
b1,b2,b3 = [],[],[]
mode_used = "FRONT"

def show_graph():
    if not t_series:
        return

    fig,axs = plt.subplots(3,1,figsize=(12,8),sharex=True)

    axs[0].plot(t_series,v1,color="red")
    axs[0].axhline(THRESH_SHOULDER,ls="--",alpha=0.5)
    axs[0].axhline(-THRESH_SHOULDER,ls="--",alpha=0.5)
    axs[0].set_title("Metric 1 – Shoulder / Neck")

    axs[1].plot(t_series,v2,color="orange")
    axs[1].axhline(THRESH_NECK,ls="--",alpha=0.5)
    axs[1].axhline(-THRESH_NECK,ls="--",alpha=0.5)
    axs[1].set_title("Metric 2 – Neck / Torso")

    axs[2].plot(t_series,v3,color="purple")
    axs[2].axhline(THRESH_BODY,ls="--",alpha=0.5)
    axs[2].axhline(-THRESH_BODY,ls="--",alpha=0.5)
    axs[2].set_title("Metric 3 – Body Axis")

    for i in range(len(t_series)-1):
        if b1[i]: axs[0].axvspan(t_series[i],t_series[i+1],color="red",alpha=0.08)
        if b2[i]: axs[1].axvspan(t_series[i],t_series[i+1],color="orange",alpha=0.08)
        if b3[i]: axs[2].axvspan(t_series[i],t_series[i+1],color="purple",alpha=0.08)

    axs[2].set_xlabel("Time (s)")
    fig.suptitle(f"{mode_used} VIEW Posture Deviation")
    plt.tight_layout()
    plt.show()

# ________________ STATE

DETECTION_RESULT=None
def save_result(r,_,__):
    global DETECTION_RESULT
    DETECTION_RESULT=r

mode="FRONT"
calibrated=False
frames=0
start_time=None

cal1,cal2,cal3=[],[],[]
base1=base2=base3=0

hist1,hist2,hist3=[],[],[]

#________________ MAIN

font=cv2.FONT_HERSHEY_SIMPLEX

if __name__=="__main__":

    options=vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=save_result
    )

    detector=vision.PoseLandmarker.create_from_options(options)
    cap=cv2.VideoCapture(0)
    fps=int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # ______________ PERFORMANCE: INITIALIZE
    process = Process(os.getpid())
    prev_frame_time = 0

    while True:
               # ____ PERFORMANCE: START TIMER
        start_time_ms = time.time() * 1000
        ok,img=cap.read()
        if not ok:
            continue

        h,w=img.shape[:2]

        draw_panel(img,10,10,300,120)
        cv2.putText(img,"POSTURE MONITOR",(20,35),font,0.8,(0,255,255),2)
        cv2.putText(img,f"MODE: {mode}",(20,65),font,0.6,(255,255,255),2)

        # ✅ CHANGED HERE
        cv2.putText(img,"F: Front View | S: Side View",(20,90),font,0.5,(230,230,230),1)
        cv2.putText(img,"Q: Quit",(20,110),font,0.5,(230,230,230),1)

        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        detector.detect_async(mp_img,time.time_ns()//1_000_000)

        if DETECTION_RESULT and DETECTION_RESULT.pose_landmarks:
            lm=DETECTION_RESULT.pose_landmarks[0]

            bad=False
            if b1 and b2 and b3:
                bad=any([b1[-1],b2[-1],b3[-1]])

            draw_skeleton(img,lm,w,h,posture_color(bad),mode)

            if mode=="FRONT":
                l_sh=(int(lm[11].x*w),int(lm[11].y*h))
                r_sh=(int(lm[12].x*w),int(lm[12].y*h))
                l_hip=(int(lm[23].x*w),int(lm[23].y*h))
                r_hip=(int(lm[24].x*w),int(lm[24].y*h))
                neck=(int(lm[0].x*w),int(lm[0].y*h))

                m1=smooth(shoulder_tilt(l_sh,r_sh),hist1)
                m2=smooth(neck_offset(neck,l_sh,r_sh),hist2)
                m3=smooth(body_axis(l_sh,r_sh,l_hip,r_hip),hist3)
            else:
                left_vis=lm[7].visibility+lm[11].visibility+lm[23].visibility
                right_vis=lm[8].visibility+lm[12].visibility+lm[24].visibility
                ear,sh,hip=(7,11,23) if left_vis>right_vis else (8,12,24)

                ear_p=(int(lm[ear].x*w),int(lm[ear].y*h))
                sh_p=(int(lm[sh].x*w),int(lm[sh].y*h))
                hip_p=(int(lm[hip].x*w),int(lm[hip].y*h))

                m1=smooth(findAngle(*sh_p,*ear_p),hist1)
                m2=smooth(findAngle(*hip_p,*sh_p),hist2)
                m3=0

            if not calibrated:
                cal1.append(m1); cal2.append(m2); cal3.append(m3)
                frames+=1
                elapsed=frames/fps

                draw_panel(img,10,h-90,300,70)
                cv2.putText(img,"Sit upright 5s to calibrate",(20,h-60),font,0.6,(0,255,255),2)
                draw_progress(img,20,h-45,260,15,min(elapsed/CALIBRATION_DURATION,1))

                if elapsed>=CALIBRATION_DURATION:
                    base1=sum(cal1)/len(cal1)
                    base2=sum(cal2)/len(cal2)
                    base3=sum(cal3)/len(cal3)
                    calibrated=True
                    start_time=time.time()
                    mode_used=mode
            else:
                d1=deadzone(m1-base1)
                d2=deadzone(m2-base2)
                d3=deadzone(m3-base3)

                t=time.time()-start_time
                t_series.append(t)
                v1.append(d1); v2.append(d2); v3.append(d3)

                if mode=="FRONT":
                    b1.append(abs(d1)>THRESH_SHOULDER)
                    b2.append(abs(d2)>THRESH_NECK)
                    b3.append(abs(d3)>THRESH_BODY)
                else:
                    b1.append(abs(d1)>THRESH_NECK_ANGLE)
                    b2.append(abs(d2)>THRESH_TORSO_ANGLE)
                    b3.append(False)

          # _______ PERFORMANCE: END TIMER
        end_time_ms = time.time() * 1000
        execution_time = end_time_ms - start_time_ms  # ms

        # RAM usage (MB)
        memory_usage = process.memory_info().rss / 1024 / 1024

        # FPS calculation
        new_frame_time = time.time()
        if prev_frame_time != 0:
            fps_perf = 1 / (new_frame_time - prev_frame_time)
        else:
            fps_perf = 0
        prev_frame_time = new_frame_time

        # Print objective data (FOR EXPERIMENTS & EVALUATION)
        print(
            f"FPS: {int(fps_perf)} | "
            f"Time: {execution_time:.2f} ms | "
            f"RAM: {memory_usage:.2f} MB"
        )

        cv2.imshow("Posture Monitor",img)
        key=cv2.waitKey(5)&0xFF

        if key==ord('f'):
            mode="FRONT"; calibrated=False; frames=0
            cal1.clear(); cal2.clear(); cal3.clear()
            hist1.clear(); hist2.clear(); hist3.clear()
        if key==ord('s'):
            mode="SIDE"; calibrated=False; frames=0
            cal1.clear(); cal2.clear(); cal3.clear()
            hist1.clear(); hist2.clear(); hist3.clear()
        if key==ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    show_graph()
