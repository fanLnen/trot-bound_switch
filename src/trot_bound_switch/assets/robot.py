"""SA16 articulation configuration."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from trot_bound_switch.paths import URDF_PATH


SA16_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF_PATH),
        force_usd_conversion=True,
        merge_fixed_joints=False,
        fix_base=False,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.0,
            ".*_calf_joint": 0.0,
            ".*_ankle_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=[f".*_{name}_joint"],
            effort_limit=48.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
            friction=0.0,
        )
        for name in ("hip", "thigh", "calf", "ankle")
    },
)
