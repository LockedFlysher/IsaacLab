# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""GUI + auto-tuning loop for Hector wholebody MPC base-height tracking.

功能目标：
- 生成默认高度上下 0.05 m 的 1 Hz 正弦高度参考，单次试验 10 s。
- 倾斜（滚转/俯仰）超过 20 度判为失败。
- 提供简单 GUI：参数列表输入、开始/停止、显示最新/最佳结果和高度跟踪曲线。
- 自动调参：对用户输入的参数列表做笛卡尔积逐轮试验，记录 CSV，保存当前最佳。
"""

import csv
import itertools
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from isaaclab_nhb import ISAACLAB_ROBOT_DESCRIPTION_PATH
from isaaclab_nhb.envs.mpc_env.controllers.mpx import mpc_wrapper
from isaaclab_nhb.envs.mpc_env.controllers.mpx.examples.biped.config_hector_wb import Hector_MPC_CFG

import tkinter as tk
from tkinter import ttk

import matplotlib

# TkAgg 后端以支持嵌入式画布
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


LOG_DIR = os.path.join(os.path.dirname(__file__), "tuning_logs")
RUNS_CSV = os.path.join(LOG_DIR, "hector_wb_runs.csv")
BEST_CSV = os.path.join(LOG_DIR, "hector_wb_best.csv")


def quat_to_rpy_deg(quat: Iterable[float]) -> Tuple[float, float, float]:
    """Convert w, x, y, z quaternion to roll, pitch, yaw in degrees."""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = 1.0 if sinp > 1.0 else sinp
    sinp = -1.0 if sinp < -1.0 else sinp
    pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def ensure_logs():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(RUNS_CSV):
        with open(RUNS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "kp_joint",
                    "kd_joint",
                    "tau_ff_scale",
                    "pd_scale",
                    "rmse_height",
                    "max_tilt_deg",
                    "succeeded",
                    "failure_time",
                    "run_duration",
                ]
            )
    if not os.path.exists(BEST_CSV):
        with open(BEST_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "kp_joint",
                    "kd_joint",
                    "tau_ff_scale",
                    "pd_scale",
                    "rmse_height",
                    "max_tilt_deg",
                    "succeeded",
                    "failure_time",
                    "run_duration",
                ]
            )


@dataclass
class TuningResult:
    params: Dict[str, float]
    rmse_height: float
    max_tilt_deg: float
    succeeded: bool
    failure_time: float
    run_duration: float
    trace_time: List[float]
    trace_target: List[float]
    trace_height: List[float]
    trace_tilt: List[float]


class HectorTuner:
    def __init__(
        self,
        sim_frequency: float = 200.0,
        mpc_frequency: float = 50.0,
        amplitude: float = 0.05,
        reference_frequency: float = 1.0,
        episode_length: float = 10.0,
    ):
        self.n_env = 1
        self.sim_frequency = sim_frequency
        self.mpc_frequency = mpc_frequency
        self.amplitude = amplitude
        self.reference_frequency = reference_frequency
        self.episode_length = episode_length
        self.mpc_stride = int(self.sim_frequency / self.mpc_frequency)

        # Build model, data, and MPC
        model = mujoco.MjModel.from_xml_path(
            os.path.join(ISAACLAB_ROBOT_DESCRIPTION_PATH, "Hector", "mjcf", "scene_Hector.xml")
        )
        data = mujoco.MjData(model)
        model.opt.timestep = 1.0 / self.sim_frequency

        self.mpc_cfg = Hector_MPC_CFG
        self.mpc_cfg.n_envs = self.n_env
        self.mpc = mpc_wrapper.batched_legged_robot_mpc_jax_core.BatchedLeggedRobotMPCJAXCore(
            self.mpc_cfg, self.n_env
        )
        self.batch_mpc_data = jax.vmap(lambda _: self.mpc.make_data())(jnp.arange(self.n_env))

        self.base_height_nominal = float(self.mpc.robot.default_configuration[2])

        # Jitted helpers
        def _solve_mpc(mpc_data, q, dq, command):
            mpc_data, tau, q_mpc, _ = self.mpc.run(mpc_data, q, dq, command)
            return mpc_data, tau, q_mpc

        self.solve_mpc = jax.jit(jax.vmap(_solve_mpc))

        def _mjx_step(m, d, action):
            tau_dof = jnp.zeros_like(d.qfrc_applied)
            tau_dof = tau_dof.at[6 : 6 + self.mpc.robot.n_joints].set(action)
            d = d.replace(ctrl=jnp.zeros_like(d.ctrl), qfrc_applied=tau_dof)
            return mjx.step(m, d)

        self.step = jax.jit(jax.vmap(_mjx_step, in_axes=(None, 0, 0)))

        def _set_inputs_helper(d, command, base_height):
            q = d.qpos
            dq = d.qvel
            mpc_command = jnp.array([command[0], command[1], 0.0, 0.0, 0.0, command[2], base_height])
            return q, dq, mpc_command

        self.set_inputs = jax.jit(jax.vmap(_set_inputs_helper))

        # Initial batched data
        self.mjx_model = mjx.put_model(model)
        self.mjx_data = mjx.put_data(model, data)
        qpos0 = jnp.tile(jnp.concatenate([self.mpc.robot.default_configuration]), (self.n_env, 1))
        self.qpos0 = qpos0
        self.action = jnp.zeros((self.n_env, self.mpc.robot.n_joints))
        self.batch_command = jnp.zeros((self.n_env, 3))
        self.reset_state()

    def reset_state(self):
        self.batch_data = jax.vmap(
            lambda x: self.mjx_data.replace(
                qpos=x, qvel=jnp.zeros(6 + self.mpc.robot.n_joints), ctrl=jnp.zeros(self.mpc.robot.n_joints)
            )
        )(self.qpos0)
        self.batch_mpc_data = self.mpc.reset(self.batch_mpc_data, jnp.arange(self.n_env))
        self.action = jnp.zeros((self.n_env, self.mpc.robot.n_joints))

    def run_trial(
        self,
        params: Dict[str, float],
        stop_event: Optional[threading.Event] = None,
        record_trace: bool = True,
    ) -> TuningResult:
        kp_joint = params.get("kp_joint", 20.0)
        kd_joint = params.get("kd_joint", 0.5)
        tau_ff_scale = params.get("tau_ff_scale", 1.0)
        pd_scale = params.get("pd_scale", 0.2)

        self.reset_state()
        total_steps = int(self.episode_length * self.sim_frequency)
        error_acc = 0.0
        samples = 0
        max_tilt = 0.0
        failure_time = 0.0
        succeeded = True
        trace_t = []
        trace_target = []
        trace_height = []
        trace_tilt = []

        for step_idx in range(total_steps):
            if stop_event and stop_event.is_set():
                break

            t = step_idx / self.sim_frequency
            base_target = self.base_height_nominal + self.amplitude * math.sin(
                2 * math.pi * self.reference_frequency * t
            )
            base_target_vec = jnp.full((self.n_env,), base_target)

            if step_idx % self.mpc_stride == 0:
                batch_q, batch_dq, batch_mpc_command = self.set_inputs(
                    self.batch_data, self.batch_command, base_target_vec
                )
                self.batch_mpc_data, tau_ff, q_des = self.solve_mpc(
                    self.batch_mpc_data, batch_q, batch_dq, batch_mpc_command
                )
                tau_ff.block_until_ready()
                joint_pos = batch_q[:, 7:]
                joint_vel = batch_dq[:, 6:]
                self.action = tau_ff_scale * tau_ff + pd_scale * (
                    kp_joint * (q_des - joint_pos) - kd_joint * joint_vel
                )
                self.action = jnp.clip(self.action, self.mpc.robot.min_torque, self.mpc.robot.max_torque)

            self.batch_data = self.step(self.mjx_model, self.batch_data, self.action)

            base_height = float(self.batch_data.qpos[0, 2])
            quat = [float(x) for x in self.batch_data.qpos[0, 3:7]]
            roll_deg, pitch_deg, _ = quat_to_rpy_deg(quat)
            tilt = max(abs(roll_deg), abs(pitch_deg))
            max_tilt = max(max_tilt, tilt)

            if record_trace and step_idx % 2 == 0:
                trace_t.append(t)
                trace_target.append(base_target)
                trace_height.append(base_height)
                trace_tilt.append(tilt)

            error_acc += (base_height - base_target) ** 2
            samples += 1

            if tilt > 20.0:
                succeeded = False
                failure_time = t
                break

        duration = samples / self.sim_frequency if self.sim_frequency > 0 else 0.0
        rmse = math.sqrt(error_acc / samples) if samples > 0 else float("inf")
        return TuningResult(
            params=params,
            rmse_height=rmse,
            max_tilt_deg=max_tilt,
            succeeded=succeeded,
            failure_time=failure_time,
            run_duration=duration,
            trace_time=trace_t,
            trace_target=trace_target,
            trace_height=trace_height,
            trace_tilt=trace_tilt,
        )


class TuningGUI:
    def __init__(self):
        ensure_logs()
        self.tuner = HectorTuner()
        self.root = tk.Tk()
        self.root.title("Hector WB MPC Auto Tuning")
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.best_result: Optional[TuningResult] = None
        self.update_queue: "queue.Queue[TuningResult]" = queue.Queue()

        # Controls
        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="nw")

        self.param_entries = {
            "kp_joint": self._add_labeled_entry(controls, "kp_joint list", "20,30"),
            "kd_joint": self._add_labeled_entry(controls, "kd_joint list", "0.5,1.0"),
            "tau_ff_scale": self._add_labeled_entry(controls, "tau_ff_scale list", "1.0"),
            "pd_scale": self._add_labeled_entry(controls, "pd_scale list", "0.2,0.3"),
        }

        self.amp_entry = self._add_labeled_entry(controls, "Amplitude (m)", "0.05")
        self.freq_entry = self._add_labeled_entry(controls, "Sine freq (Hz)", "1.0")
        self.duration_entry = self._add_labeled_entry(controls, "Run time (s)", "10.0")

        buttons = ttk.Frame(controls)
        buttons.grid(row=6, column=0, columnspan=2, pady=6, sticky="w")
        ttk.Button(buttons, text="Start Auto Tune", command=self.start).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Stop", command=self.stop).grid(row=0, column=1, padx=4)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.status_var, foreground="blue").grid(row=7, column=0, columnspan=2, sticky="w")
        self.best_var = tk.StringVar(value="Best: N/A")
        ttk.Label(controls, textvariable=self.best_var, foreground="green").grid(row=8, column=0, columnspan=2, sticky="w")

        # Plot
        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Base height tracking")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Height (m)")
        self.height_line, = self.ax.plot([], [], label="height")
        self.target_line, = self.ax.plot([], [], label="target")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

        self.root.after(200, self._poll_updates)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, default: str) -> ttk.Entry:
        row = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="we", pady=2)
        parent.grid_columnconfigure(1, weight=1)
        return entry

    def parse_list(self, text: str) -> List[float]:
        vals = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                vals.append(float(chunk))
            except ValueError:
                continue
        return vals

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        try:
            amplitude = float(self.amp_entry.get())
            frequency = float(self.freq_entry.get())
            duration = float(self.duration_entry.get())
        except ValueError:
            self.status_var.set("Invalid amplitude/frequency/duration")
            return

        kp_list = self.parse_list(self.param_entries["kp_joint"].get())
        kd_list = self.parse_list(self.param_entries["kd_joint"].get())
        tau_list = self.parse_list(self.param_entries["tau_ff_scale"].get())
        pd_list = self.parse_list(self.param_entries["pd_scale"].get())
        if not kp_list or not kd_list or not tau_list or not pd_list:
            self.status_var.set("Parameter lists cannot be empty")
            return

        self.tuner.amplitude = amplitude
        self.tuner.reference_frequency = frequency
        self.tuner.episode_length = duration

        combos = list(itertools.product(kp_list, kd_list, tau_list, pd_list))
        self.status_var.set(f"Running {len(combos)} combinations...")
        self.worker = threading.Thread(
            target=self._worker_loop,
            args=(combos,),
            daemon=True,
        )
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping after current step...")

    def _worker_loop(self, combos: List[Tuple[float, float, float, float]]):
        for idx, (kp, kd, tau_ff_scale, pd_scale) in enumerate(combos, start=1):
            if self.stop_event.is_set():
                break
            params = {
                "kp_joint": kp,
                "kd_joint": kd,
                "tau_ff_scale": tau_ff_scale,
                "pd_scale": pd_scale,
            }
            start_time = time.time()
            result = self.tuner.run_trial(params, stop_event=self.stop_event, record_trace=True)
            elapsed = time.time() - start_time
            self._log_run(result)
            self._update_best(result)
            self.status_var.set(
                f"Run {idx}/{len(combos)} done in {elapsed:.1f}s | RMSE={result.rmse_height:.4f} | tilt={result.max_tilt_deg:.1f}° | success={result.succeeded}"
            )
            self.update_queue.put(result)

        self.status_var.set("Idle")
        self.stop_event.clear()

    def _log_run(self, result: TuningResult):
        with open(RUNS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.time(),
                    result.params.get("kp_joint"),
                    result.params.get("kd_joint"),
                    result.params.get("tau_ff_scale"),
                    result.params.get("pd_scale"),
                    result.rmse_height,
                    result.max_tilt_deg,
                    int(result.succeeded),
                    result.failure_time,
                    result.run_duration,
                ]
            )

    def _update_best(self, result: TuningResult):
        if (self.best_result is None) or (
            result.succeeded
            and (not self.best_result.succeeded or result.rmse_height < self.best_result.rmse_height)
        ):
            self.best_result = result
            with open(BEST_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "kp_joint",
                        "kd_joint",
                        "tau_ff_scale",
                        "pd_scale",
                        "rmse_height",
                        "max_tilt_deg",
                        "succeeded",
                        "failure_time",
                        "run_duration",
                    ]
                )
                writer.writerow(
                    [
                        time.time(),
                        result.params.get("kp_joint"),
                        result.params.get("kd_joint"),
                        result.params.get("tau_ff_scale"),
                        result.params.get("pd_scale"),
                        result.rmse_height,
                        result.max_tilt_deg,
                        int(result.succeeded),
                        result.failure_time,
                        result.run_duration,
                    ]
                )
            self.best_var.set(
                f"Best RMSE {result.rmse_height:.4f} | tilt {result.max_tilt_deg:.1f}° | params {result.params}"
            )

    def _poll_updates(self):
        try:
            while True:
                result = self.update_queue.get_nowait()
                self._update_plot(result)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_updates)

    def _update_plot(self, result: TuningResult):
        self.height_line.set_data(result.trace_time, result.trace_height)
        self.target_line.set_data(result.trace_time, result.trace_target)
        if result.trace_time:
            self.ax.set_xlim(result.trace_time[0], result.trace_time[-1])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def run(self):
        self.root.mainloop()


def main():
    gui = TuningGUI()
    gui.run()


if __name__ == "__main__":
    main()
