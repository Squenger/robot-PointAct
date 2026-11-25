import numpy as np
from collections import deque


class AdaptiveHorizonMonitor:
    """
    Monitors visual/3D depth and proprioceptive signals at each timestep
    to dynamically interrupt open-loop action chunks and trigger early replanning.
    """

    def __init__(
        self,
        mode: str = "fixed",
        delta_action: bool = True,
        depth_pixel_thresh: float = 0.025,
        depth_ratio_thresh: float = 0.010,
        depth_spike_factor: float = 2.5,
        cos_sim_thresh: float = -0.10,
        cooldown_steps: int = 3,
    ):
        assert mode in ["fixed", "adaptive"]
        self.mode = mode
        self.delta_action = delta_action
        self.depth_pixel_thresh = depth_pixel_thresh
        self.depth_ratio_thresh = depth_ratio_thresh
        self.depth_spike_factor = depth_spike_factor
        self.cos_sim_thresh = cos_sim_thresh
        self.cooldown_steps = cooldown_steps

        self.reset()

    def reset(self):
        self.ema_depth_ratio = None
        self.cooldown_remaining = 0
        self.step_count = 0
        self.gripper_state = None  # -1 (open) or +1 (closed)
        self.last_gripper_trigger_step = -999
        self.consec_anomaly = 0

    def _check_depth_surprise(self, obs: dict, prev_obs: dict) -> bool:
        if "observation.depths.front" not in obs or "observation.depths.front" not in prev_obs:
            return False

        d_curr = obs["observation.depths.front"]
        d_prev = prev_obs["observation.depths.front"]
        valid = (d_curr > 0.05) & (d_prev > 0.05) & (d_curr < 3.0) & (d_prev < 3.0)
        if not np.any(valid):
            return False

        diff = np.abs(d_curr[valid] - d_prev[valid])
        changed_ratio = float(np.mean(diff > self.depth_pixel_thresh))

        triggered = False
        if self.ema_depth_ratio is None:
            self.ema_depth_ratio = changed_ratio
        else:
            if self.step_count >= 5 and changed_ratio > max(
                self.depth_ratio_thresh, self.depth_spike_factor * self.ema_depth_ratio
            ):
                triggered = True
            alpha = 0.15 if not triggered else 0.02
            self.ema_depth_ratio = (1.0 - alpha) * self.ema_depth_ratio + alpha * changed_ratio

        return triggered

    def _check_proprio_anomaly(self, obs: dict, prev_obs: dict, last_action: np.ndarray) -> str | None:
        if last_action is None or not self.delta_action:
            self.consec_anomaly = 0
            return None

        d_eef = obs["observation.state"][:3] - prev_obs["observation.state"][:3]
        cmd_xyz = last_action[:3]
        cmd_norm = float(np.linalg.norm(cmd_xyz))
        eef_norm = float(np.linalg.norm(d_eef))

        is_anomaly = False
        reason = None

        # Strong command but arm moves in opposite/orthogonal direction (e.g. drift/noise)
        if cmd_norm > 0.25 and eef_norm > 0.002:
            cos_sim = float(np.dot(d_eef, cmd_xyz) / (eef_norm * cmd_norm + 1e-8))
            if cos_sim < self.cos_sim_thresh:
                is_anomaly = True
                reason = "proprio_anomaly"
        # Very strong command but robot is completely blocked
        elif cmd_norm > 0.50 and eef_norm < 0.0002:
            is_anomaly = True
            reason = "proprio_stall"

        if is_anomaly:
            self.consec_anomaly += 1
            if self.consec_anomaly >= 2:
                self.consec_anomaly = 0
                return reason
        else:
            self.consec_anomaly = 0

        return None

    def _check_gripper_transition(self, last_action: np.ndarray | None) -> bool:
        if last_action is None:
            return False

        g_val = float(last_action[-1])
        # Hysteresis: only commit to open (-1) or closed (+1) when signal is clear
        if g_val > 0.4:
            new_state = 1
        elif g_val < -0.4:
            new_state = -1
        else:
            return False

        if self.gripper_state is None:
            self.gripper_state = new_state
            return False

        if new_state != self.gripper_state and (self.step_count - self.last_gripper_trigger_step) >= 10:
            self.gripper_state = new_state
            self.last_gripper_trigger_step = self.step_count
            return True

        return False

    def should_replan(
        self,
        obs: dict,
        prev_obs: dict | None,
        last_action: np.ndarray | None,
    ) -> tuple[bool, str | None]:
        """Returns (should_replan_now, reason_str)."""
        self.step_count += 1

        if self.mode == "fixed" or prev_obs is None:
            return False, None

        # 1. Visual/3D depth surprise (instant object shift detection)
        if self._check_depth_surprise(obs, prev_obs):
            self.cooldown_remaining = self.cooldown_steps
            return True, "depth_surprise"

        # 2. Confirmed proprioceptive anomaly (action drift / collision)
        proprio_reason = self._check_proprio_anomaly(obs, prev_obs, last_action)
        if proprio_reason is not None:
            self.cooldown_remaining = self.cooldown_steps
            return True, proprio_reason

        # 3. Clean gripper open/close transition
        if self._check_gripper_transition(last_action):
            self.cooldown_remaining = max(self.cooldown_remaining, 2)
            return True, "gripper_transition"

        # 4. Short post-event closed-loop window
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return True, "cooldown"

        return False, None
    




class BacktrackController:
    """Détecte une saisie à vide via gripper_qpos et exécute un retour arrière contrôlé."""

    def __init__(
        self,
        enabled: bool = True,
        empty_grasp_width: float = 0.008,
        close_confirm_steps: int = 8,
        open_steps: int = 4,
        retract_steps: int = 10,
        max_backtracks: int = 2,
        retract_z_bias: float = 0.35,
    ):
        self.enabled = enabled
        self.empty_grasp_width = empty_grasp_width
        self.close_confirm_steps = close_confirm_steps
        self.open_steps = open_steps
        self.retract_steps = retract_steps
        self.max_backtracks = max_backtracks
        self.retract_z_bias = retract_z_bias
        self.reset()

    def reset(self):
        self.action_history = deque(maxlen=max(1, self.retract_steps) + 10)
        self.closing_counter = 0
        self.was_open = True
        self.backtrack_queue = deque()
        self.backtrack_count = 0
        self.post_backtrack_grace = 0

    @property
    def is_active(self) -> bool:
        return len(self.backtrack_queue) > 0

    def record_action(self, action: np.ndarray):
        """Enregistre chaque action exécutée par PointACT (hors phase de backtrack)."""
        self.action_history.append(np.array(action, dtype=np.float32).copy())

    def check_and_trigger(self, obs: dict, last_action: np.ndarray) -> bool:
        """Vérifie si la pince s'est refermée à vide et prépare la séquence de retour arrière."""
        if not self.enabled or self.is_active:
            return False
        if self.backtrack_count >= self.max_backtracks:
            return False

        gripper_qpos = obs.get("robot0_gripper_qpos", None)
        if gripper_qpos is None and "observation.state" in obs:
            state = np.asarray(obs["observation.state"])
            if state.shape[-1] >= 9:
                gripper_qpos = state[7:9]
        if gripper_qpos is None:
            return False

        # Écartement réel total des deux doigts (en mètres)
        width = float(np.sum(np.abs(gripper_qpos)))
        gripper_cmd = float(last_action[-1])

        if self.post_backtrack_grace > 0:
            self.post_backtrack_grace -= 1
            self.closing_counter = 0
            if width > 0.03:
                self.was_open = True
            return False

        # En convention LIBERO/PointACT : > 0 = fermer, < 0 = ouvrir
        if gripper_cmd > 0.2:
            self.closing_counter += 1
        else:
            self.closing_counter = 0
            if width > 0.03:
                self.was_open = True

        # Déclenchement : la pince était ouverte, a commandé une fermeture soutenue,
        # mais les doigts se sont refermés complètement dans le vide.
        if self.was_open and self.closing_counter >= self.close_confirm_steps:
            if width < self.empty_grasp_width:
                self._build_backtrack_plan(len(last_action))
                self.backtrack_count += 1
                self.closing_counter = 0
                self.was_open = False
                self.post_backtrack_grace = 20
                return True
            else:
                # Saisie réussie (un objet bloque les doigts)
                self.was_open = False
                self.closing_counter = 0

        return False

    def _build_backtrack_plan(self, action_dim: int):
        """Construit la trajectoire de récupération : ouverture pince + remontée/recul."""
        self.backtrack_queue.clear()

        # Phase A : rouvrir la pince sur place
        open_act = np.zeros(action_dim, dtype=np.float32)
        open_act[-1] = -1.0  # ouvrir
        for _ in range(self.open_steps):
            self.backtrack_queue.append(open_act.copy())

        # Phase B : inverser les derniers déplacements XY et remonter en Z
        recent = (
            list(self.action_history)[-self.retract_steps :]
            if self.retract_steps > 0
            else []
        )
        for past_act in reversed(recent):
            rev = np.zeros(action_dim, dtype=np.float32)
            rev[0] = -float(past_act[0]) * 0.8
            rev[1] = -float(past_act[1]) * 0.8
            rev[2] = max(-float(past_act[2]), self.retract_z_bias)
            rev[-1] = -1.0  # garder la pince ouverte
            self.backtrack_queue.append(rev)

    def next_action(self) -> np.ndarray:
        return self.backtrack_queue.popleft()
