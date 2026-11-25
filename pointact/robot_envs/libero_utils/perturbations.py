import numpy as np


class LiberoPerturber:
    """Injects reproducible physical perturbations into a LIBERO/MuJoCo episode."""

    def __init__(
        self,
        perturb_type: str = "none",
        shift_mag: float = 0.05,
        trigger_dist: float = 0.18,
        min_step: int = 20,
        fallback_step_range: tuple[int, int] = (35, 65),
        action_noise_std: float = 0.30,
        action_noise_duration: int = 6,
    ):
        assert perturb_type in ["none", "object_shift", "action_noise", "combined"]
        self.perturb_type = perturb_type
        self.shift_mag = shift_mag
        self.trigger_dist = trigger_dist
        self.min_step = min_step
        self.fallback_step_range = fallback_step_range
        self.action_noise_std = action_noise_std
        self.action_noise_duration = action_noise_duration

        self.rng = None
        self.shift_done = False
        self.fallback_shift_step = None
        self.noise_start_step = None
        self.noise_bias = None

    def reset(self, episode_seed: int):
        self.rng = np.random.default_rng(episode_seed)
        self.shift_done = False
        self.fallback_shift_step = int(
            self.rng.integers(self.fallback_step_range[0], self.fallback_step_range[1])
        )
        self.noise_start_step = int(self.rng.integers(25, 70))
        # Persistent directional drift on XYZ during the noise window
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        self.noise_bias = (direction * self.action_noise_std).astype(np.float32)

    def _find_target_free_joint(self, libero_env, eef_pos: np.ndarray):
        """Finds the movable object (free joint) closest to the robot end-effector."""
        model = libero_env.env.sim.model
        data = libero_env.env.sim.data

        best_joint = None
        best_dist = float("inf")
        for j_name in model.joint_names:
            j_id = model.joint_name2id(j_name)
            # MuJoCo mjJNT_FREE == 0 (movable 6-DoF objects on the table)
            if int(model.jnt_type[j_id]) != 0:
                continue
            qpos_adr = int(model.jnt_qposadr[j_id])
            obj_pos = data.qpos[qpos_adr : qpos_adr + 3]
            dist = float(np.linalg.norm(obj_pos - eef_pos))
            # Ignore objects already grasped inside the gripper (< 4.5 cm)
            if 0.045 < dist < best_dist:
                best_dist = dist
                best_joint = (j_name, j_id, qpos_adr, int(model.jnt_dofadr[j_id]))

        return best_joint, best_dist

    def _apply_object_shift(self, libero_env, eef_pos: np.ndarray):
        target, dist = self._find_target_free_joint(libero_env, eef_pos)
        if target is None:
            return None

        j_name, _, qpos_adr, qvel_adr = target
        sim = libero_env.env.sim

        # Random horizontal shift (XY plane) of exact magnitude shift_mag
        angle = float(self.rng.uniform(0, 2 * np.pi))
        dx = self.shift_mag * np.cos(angle)
        dy = self.shift_mag * np.sin(angle)

        new_x = np.clip(sim.data.qpos[qpos_adr] + dx, -0.35, 0.35)
        new_y = np.clip(sim.data.qpos[qpos_adr + 1] + dy, -0.35, 0.35)

        sim.data.qpos[qpos_adr] = new_x
        sim.data.qpos[qpos_adr + 1] = new_y
        # Zero linear & angular velocities so the object settles cleanly
        sim.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        sim.forward()

        self.shift_done = True
        return {"event": "object_shift", "joint": j_name, "dx": float(dx), "dy": float(dy), "dist": dist}

    def maybe_perturb(self, libero_env, t: int, obs: dict, action: np.ndarray):
        """
        Called right before libero_env.step(action).
        Returns (possibly_modified_action, event_dict_or_None).
        """
        if self.perturb_type == "none":
            return action, None

        event = None
        eef_pos = obs["observation.state"][:3]

        # 1. Physical object shift in MuJoCo
        if self.perturb_type in ("object_shift", "combined") and not self.shift_done:
            if t >= self.min_step:
                _, dist = self._find_target_free_joint(libero_env, eef_pos)
                if dist <= self.trigger_dist or t >= self.fallback_shift_step:
                    event = self._apply_object_shift(libero_env, eef_pos)

        # 2. Action drift / slip perturbation
        mod_action = action.copy()
        if self.perturb_type in ("action_noise", "combined"):
            if self.noise_start_step <= t < self.noise_start_step + self.action_noise_duration:
                mod_action[:3] = np.clip(mod_action[:3] + self.noise_bias, -1.0, 1.0)
                if event is None:
                    event = {"event": "action_noise", "step": t}

        return mod_action, event


PerturbationManager = LiberoPerturber


PerturbationManager = LiberoPerturber
