#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, threading, time
from collections import deque
from typing import Dict, List, Tuple

import rclpy
from geometry_msgs.msg import Accel, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64, String
from tkinter import Tk, StringVar
from tkinter import scrolledtext, ttk

from bisa.msg import MPCPerformance


def _fmt(v, d=3):
    return "-" if not math.isfinite(v) else f"{v:.{d}f}"


def _parse_gate_status(data):
    parsed = {}
    for token in data.split():
        if "=" in token:
            k, v = token.split("=", 1)
            parsed[k] = v
    return parsed


# ── CAV01 진단 임계값 ──────────────────────────────────────────────
CTE_WARN   = 0.05   # m  이상이면 경고
CTE_CRIT   = 0.12   # m  이상이면 위험
HEAD_WARN  = 0.15   # rad
HEAD_CRIT  = 0.35   # rad
VCMD_LOW   = 0.05   # m/s 이하면 사실상 정지


class RuntimeLogMonitor(Node):
    def __init__(self):
        super().__init__("cav_runtime_log_gui")

        self.declare_parameter("cav_ids", [1, 2, 3, 4])
        cav_ids_param = self.get_parameter("cav_ids").get_parameter_value().integer_array_value
        self.cav_ids = [int(v) for v in cav_ids_param] if cav_ids_param else [1, 2, 3, 4]

        self._lock = threading.Lock()
        self._events = deque(maxlen=800)

        self.offline_status = "N/A"
        self.offline_status_time = 0.0

        self.state = {}
        for cid in self.cav_ids:
            self.state[cid] = {
                "mode": "N/A", "block": -1,
                "raw_v": float("nan"), "sim_v": float("nan"),
                "cap": float("nan"), "flags": "",
                "solver": "N/A", "updated": 0.0,
                "raw_zero_pass_latched": False,
                "sim_zero_pass_latched": False,
            }

        # ── CAV01 전용 진단 상태 ─────────────────────────────────
        self.diag = {
            "cte":        float("nan"),   # 현재 횡방향 오차 [m]
            "head_err":   float("nan"),   # 현재 heading 오차 [rad]
            "v_cmd":      float("nan"),   # 명령 속도
            "w_cmd":      float("nan"),   # 명령 각속도
            "kappa_r":    float("nan"),   # 경로 곡률
            "cte_max":    0.0,            # 이번 랩 최대 CTE
            "cte_spike_cnt": 0,           # CTE_CRIT 초과 횟수
            "stall_cnt":  0,              # 정지 카운트
            "last_guard": "none",         # 마지막 활성 guard
            "guard_hist": deque(maxlen=6),# 최근 guard 이력
        }
        self._pose = None
        self._local_path = []

        qos_re  = QoSProfile(depth=30, reliability=ReliabilityPolicy.RELIABLE)
        qos_be  = QoSProfile(depth=30, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_lat = QoSProfile(depth=10,  reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(String, "/offline_scheduler/status",
                                 self._offline_status_cb, qos_lat)

        # ── CAV01 진단 전용 구독 ─────────────────────────────────
        self.create_subscription(PoseStamped, "/CAV_01",
                                 self._pose_cb, qos_be)
        self.create_subscription(Path, "/cav01/local_path",
                                 self._path_cb, qos_be)
        self.create_subscription(Accel, "/CAV_01_accel",
                                 self._accel_diag_cb, qos_be)

        # ── 공통 구독 ────────────────────────────────────────────
        self._subs = []
        for cid in self.cav_ids:
            s = f"{cid:02d}"
            for topic, cls, cb in [
                (f"/cav{s}/priority_gate_status", String,
                 lambda m, c=cid: self._gate_cb(c, m)),
                (f"/cav{s}/accel_raw",  Accel,
                 lambda m, c=cid: self._raw_cb(c, m)),
                (f"/sim/cav{s}/accel",  Accel,
                 lambda m, c=cid: self._sim_cb(c, m)),
                (f"/cav{s}/offline_speed_cap", Float64,
                 lambda m, c=cid: self._cap_cb(c, m)),
                (f"/cav{s}/mpc_performance", MPCPerformance,
                 lambda m, c=cid: self._perf_cb(c, m)),
            ]:
                self._subs.append(
                    self.create_subscription(cls, topic, cb, qos_re))

        # 20 Hz 진단 타이머
        self.create_timer(0.05, self._diag_tick)

    # ── CAV01 전용 콜백 ───────────────────────────────────────────
    def _pose_cb(self, msg):
        self._pose = msg

    def _path_cb(self, msg):
        self._local_path = msg.poses

    def _accel_diag_cb(self, msg):
        with self._lock:
            self.diag["v_cmd"] = float(msg.linear.x)
            self.diag["w_cmd"] = float(msg.angular.z)

    def _diag_tick(self):
        """20 Hz: CTE / heading_err / kappa_r 계산 + 임계값 초과 로깅"""
        if self._pose is None or len(self._local_path) < 3:
            return

        ex = self._pose.pose.position.x
        ey = self._pose.pose.position.y

        # 가장 가까운 경로점
        min_d, closest = float("inf"), 0
        for i, ps in enumerate(self._local_path):
            d = math.hypot(ps.pose.position.x - ex, ps.pose.position.y - ey)
            if d < min_d:
                min_d, closest = d, i

        cte, head_err, kappa_r = 0.0, 0.0, 0.0
        if closest + 1 < len(self._local_path):
            p0 = self._local_path[closest].pose.position
            p1 = self._local_path[closest + 1].pose.position
            path_yaw = math.atan2(p1.y - p0.y, p1.x - p0.x)
            rx, ry = ex - p0.x, ey - p0.y
            cte = -math.sin(path_yaw) * rx + math.cos(path_yaw) * ry

            q = self._pose.pose.orientation
            norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
            if abs(norm - 1.0) > 0.15:
                yaw = q.z
            else:
                siny = 2.0*(q.w*q.z + q.x*q.y)
                cosy = 1.0 - 2.0*(q.y**2 + q.z**2)
                yaw = math.atan2(siny, cosy)
            head_err = path_yaw - yaw
            while head_err >  math.pi: head_err -= 2*math.pi
            while head_err < -math.pi: head_err += 2*math.pi

        idx = min(closest, len(self._local_path) - 3)
        x1 = self._local_path[idx].pose.position.x
        y1 = self._local_path[idx].pose.position.y
        x2 = self._local_path[idx+1].pose.position.x
        y2 = self._local_path[idx+1].pose.position.y
        x3 = self._local_path[idx+2].pose.position.x
        y3 = self._local_path[idx+2].pose.position.y
        a = math.hypot(x2-x1, y2-y1)
        b = math.hypot(x3-x2, y3-y2)
        c = math.hypot(x3-x1, y3-y1)
        area2 = abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
        if area2 > 1e-9 and a > 1e-6 and b > 1e-6 and c > 1e-6:
            kappa_r = 2.0*area2/(a*b*c)

        cte_abs = abs(cte)
        head_abs = abs(head_err)

        with self._lock:
            d = self.diag
            d["cte"] = cte
            d["head_err"] = head_err
            d["kappa_r"] = kappa_r

            # 최대 CTE 갱신
            if cte_abs > d["cte_max"]:
                d["cte_max"] = cte_abs

            v_cmd = d["v_cmd"]
            w_cmd = d["w_cmd"]

            # guard 판별
            guard = "none"
            if cte_abs > CTE_CRIT:
                guard = "OFF_PATH"
                d["cte_spike_cnt"] += 1
            elif head_abs > HEAD_CRIT:
                guard = "HEADING_CRIT"
            elif cte_abs > CTE_WARN and head_abs > HEAD_WARN:
                guard = "CTE+HEAD_WARN"
            elif abs(kappa_r) > 0.30:
                guard = "SHARP_CURVE"
            elif cte_abs > CTE_WARN:
                guard = "CTE_WARN"
            elif head_abs > HEAD_WARN:
                guard = "HEAD_WARN"

            if math.isfinite(v_cmd) and v_cmd < VCMD_LOW:
                d["stall_cnt"] += 1
                guard = "STALL"

            # guard 변화 시 로그
            if guard != d["last_guard"]:
                if guard != "none":
                    d["guard_hist"].append(guard)
                    self._push_event(
                        f"[CAV01] {guard} | "
                        f"CTE={_fmt(cte,4)}m  HEAD={_fmt(head_err,3)}rad  "
                        f"κ={_fmt(kappa_r,4)}  v={_fmt(v_cmd,3)}  w={_fmt(w_cmd,3)}"
                    )
                d["last_guard"] = guard

    # ── 공통 콜백 ─────────────────────────────────────────────────
    def _push_event(self, text):
        stamp = time.strftime("%H:%M:%S")
        self._events.append(f"[{stamp}] {text}")

    def _offline_status_cb(self, msg):
        now = time.monotonic()
        with self._lock:
            if msg.data != self.offline_status:
                self._push_event(f"offline_status: {msg.data}")
            self.offline_status = msg.data
            self.offline_status_time = now

    def _gate_cb(self, cid, msg):
        kv = _parse_gate_status(msg.data)
        now = time.monotonic()
        with self._lock:
            s = self.state[cid]
            prev_mode = str(s["mode"])
            mode = kv.get("mode", prev_mode)
            try: block = int(kv.get("block", "-1"))
            except: block = -1
            s["mode"] = mode; s["block"] = block
            flags = [k for k in ["geo","res","occ","near","imm","ems","tie",
                                  "rear_slow","rear_stop","offline_hold"]
                     if kv.get(k,"0")=="1"]
            s["flags"] = "|".join(flags) if flags else "-"
            s["updated"] = now
            if mode != prev_mode:
                self._push_event(
                    f"CAV{cid:02d} mode: {prev_mode}->{mode} "
                    f"(block={block}, flags={s['flags']})")
            self._check_pass_zero_locked(cid, s)

    def _raw_cb(self, cid, msg):
        now = time.monotonic()
        with self._lock:
            s = self.state[cid]
            s["raw_v"] = max(0.0, float(msg.linear.x))
            s["updated"] = now
            self._check_pass_zero_locked(cid, s)

    def _sim_cb(self, cid, msg):
        now = time.monotonic()
        with self._lock:
            s = self.state[cid]
            s["sim_v"] = max(0.0, float(msg.linear.x))
            s["updated"] = now
            self._check_pass_zero_locked(cid, s)

    def _cap_cb(self, cid, msg):
        now = time.monotonic()
        cap = max(0.0, float(msg.data))
        with self._lock:
            s = self.state[cid]
            prev = float(s["cap"]) if math.isfinite(s["cap"]) else float("nan")
            s["cap"] = cap; s["updated"] = now
            if cap <= 1e-3 and (not math.isfinite(prev) or prev > 1e-3):
                self._push_event(f"CAV{cid:02d} offline_cap->0.000")

    def _perf_cb(self, cid, msg):
        now = time.monotonic()
        with self._lock:
            s = self.state[cid]
            prev = str(s["solver"])
            s["solver"] = msg.solver_status; s["updated"] = now
            if msg.solver_status != prev:
                self._push_event(
                    f"CAV{cid:02d} solver: {prev}->{msg.solver_status}")

    def _check_pass_zero_locked(self, cid, s):
        mode = str(s["mode"])
        raw_v = float(s["raw_v"]); sim_v = float(s["sim_v"])
        if mode != "PASS":
            s["raw_zero_pass_latched"] = s["sim_zero_pass_latched"] = False
            return
        if math.isfinite(raw_v):
            if raw_v < 0.02 and not s["raw_zero_pass_latched"]:
                self._push_event(f"CAV{cid:02d} raw_v≈0 while PASS")
                s["raw_zero_pass_latched"] = True
            elif raw_v > 0.08:
                s["raw_zero_pass_latched"] = False
        if math.isfinite(sim_v):
            if sim_v < 0.02 and not s["sim_zero_pass_latched"]:
                self._push_event(f"CAV{cid:02d} sim_v≈0 while PASS")
                s["sim_zero_pass_latched"] = True
            elif sim_v > 0.08:
                s["sim_zero_pass_latched"] = False

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            age = (now - self.offline_status_time) if self.offline_status_time > 0 else float("nan")
            offline_line = f"offline_status: {self.offline_status} (age={_fmt(age,1)}s)"
            rows = []
            for cid in self.cav_ids:
                s = self.state[cid]
                upd = float(s["updated"]); a = (now - upd) if upd > 0 else float("nan")
                rows.append((cid, str(s["mode"]), int(s["block"]),
                             float(s["raw_v"]), float(s["sim_v"]), float(s["cap"]),
                             str(s["flags"]), str(s["solver"]), a))
            d = self.diag
            diag_snap = {
                "cte":      d["cte"],
                "head_err": d["head_err"],
                "kappa_r":  d["kappa_r"],
                "v_cmd":    d["v_cmd"],
                "w_cmd":    d["w_cmd"],
                "cte_max":  d["cte_max"],
                "spikes":   d["cte_spike_cnt"],
                "stalls":   d["stall_cnt"],
                "guard":    d["last_guard"],
                "guard_hist": list(d["guard_hist"]),
            }
            events = list(self._events); self._events.clear()
            return offline_line, rows, diag_snap, events


class RuntimeLogWindow:
    def __init__(self, monitor):
        self.monitor = monitor
        self.root = Tk()
        self.root.title("CAV Runtime Log Monitor")
        self.root.geometry("1200x900")

        self.status_var = StringVar(value="offline_status: N/A")
        ttk.Label(self.root, textvariable=self.status_var,
                  font=("TkDefaultFont", 11)).pack(fill="x", padx=8, pady=(8,2))

        # ── CAV01 실시간 진단 패널 ─────────────────────────────
        diag_frame = ttk.LabelFrame(self.root, text="CAV01 실시간 진단")
        diag_frame.pack(fill="x", padx=8, pady=4)

        self.diag_vars = {}
        labels = [
            ("CTE [m]",     "cte"),
            ("Heading [rad]","head_err"),
            ("κ_r [1/m]",   "kappa_r"),
            ("v_cmd [m/s]", "v_cmd"),
            ("w_cmd [r/s]", "w_cmd"),
            ("CTE_max [m]", "cte_max"),
            ("Spikes",      "spikes"),
            ("Stalls",      "stalls"),
            ("Guard",       "guard"),
        ]
        for i, (lbl, key) in enumerate(labels):
            ttk.Label(diag_frame, text=lbl, width=14,
                      anchor="e").grid(row=i//5, column=(i%5)*2,
                                       padx=(8,2), pady=2, sticky="e")
            var = StringVar(value="-")
            self.diag_vars[key] = var
            lbl_val = ttk.Label(diag_frame, textvariable=var, width=12,
                                font=("TkFixedFont", 11), anchor="w")
            lbl_val.grid(row=i//5, column=(i%5)*2+1, padx=(0,8), pady=2)

        self.guard_hist_var = StringVar(value="")
        ttk.Label(diag_frame, text="Guard 이력:", anchor="e"
                  ).grid(row=2, column=0, padx=(8,2), pady=2, sticky="e")
        ttk.Label(diag_frame, textvariable=self.guard_hist_var,
                  font=("TkFixedFont", 10), foreground="orange"
                  ).grid(row=2, column=1, columnspan=9, sticky="w", padx=4)

        # ── CAV 상태 테이블 ───────────────────────────────────
        cols = ("cav","mode","block","raw_v","sim_v","cap","flags","solver","age_s")
        self.table = ttk.Treeview(self.root, columns=cols, show="headings", height=6)
        widths = [60,80,70,100,100,100,140,170,80]
        for col, w in zip(cols, widths):
            self.table.heading(col, text=col)
            self.table.column(col, width=w, anchor="center")
        self.table.pack(fill="x", padx=8, pady=4)
        for cid in self.monitor.cav_ids:
            self.table.insert("", "end", iid=f"cav{cid:02d}",
                              values=(cid,"-",-1,"-","-","-","-","-","-"))

        # ── 이벤트 로그 ──────────────────────────────────────
        self.log = scrolledtext.ScrolledText(self.root, wrap="word",
                                             height=22, font=("TkFixedFont", 10))
        self.log.pack(fill="both", expand=True, padx=8, pady=(4,8))
        self.log.configure(state="disabled")
        # 색상 태그
        self.log.tag_config("WARN",  foreground="#FFA500")
        self.log.tag_config("CRIT",  foreground="#FF4444")
        self.log.tag_config("INFO",  foreground="#88CCFF")

        self.root.after(200, self._refresh)

    def _append_log_lines(self, lines):
        if not lines: return
        self.log.configure(state="normal")
        for line in lines:
            tag = "INFO"
            if any(k in line for k in ["OFF_PATH","HEADING_CRIT","CRIT","stall","STALL"]):
                tag = "CRIT"
            elif any(k in line for k in ["WARN","SHARP","CTE+","mode:"]):
                tag = "WARN"
            self.log.insert("end", line+"\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _refresh(self):
        offline_line, rows, diag, events = self.monitor.snapshot()
        self.status_var.set(offline_line)

        # 진단 패널 업데이트
        cte_abs = abs(diag["cte"]) if math.isfinite(diag["cte"]) else float("nan")
        for key, var in self.diag_vars.items():
            val = diag.get(key, float("nan"))
            if isinstance(val, float):
                var.set(_fmt(val, 4) if key in ("cte","head_err","kappa_r") else _fmt(val, 3))
            else:
                var.set(str(val))
        self.guard_hist_var.set(" → ".join(diag["guard_hist"]) if diag["guard_hist"] else "-")

        # 테이블 업데이트
        for cid, mode, block, raw_v, sim_v, cap, flags, solver, age in rows:
            self.table.item(f"cav{cid:02d}", values=(
                cid, mode, block, _fmt(raw_v), _fmt(sim_v),
                _fmt(cap), flags, solver, _fmt(age,1)))

        self._append_log_lines(events)
        self.root.after(200, self._refresh)

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    monitor = RuntimeLogMonitor()
    spin_thread = threading.Thread(target=rclpy.spin, args=(monitor,), daemon=True)
    spin_thread.start()
    try:
        window = RuntimeLogWindow(monitor)
        window.run()
    finally:
        monitor.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()