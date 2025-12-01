import numpy as np
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_N5_000 as PARAM

PARAM = PARAM()


mu_void = PARAM.mu_void


def calculate_pressure_drop(
    mu: float,  # 流体动力黏度 (Pa·s)
    rho: float,  # 流体密度 (kg/m³)
    media_type: str,  # 介质类型: "颗粒"/"纤维"/"自定义"
    phi: float,  # 孔隙率 (0-1)
    L: float,  # 介质长度 (m)
    U: float = None,  # 表观流速 (m/s) (可选)
    Q: float = None,  # 体积流量 (m³/s) (可选)
    A: float = None,  # 横截面积 (m²) (需与Q一起提供)
    dp: float = None,  # 颗粒直径 (m) (介质类型为"颗粒"时必填)
    df: float = None,  # 纤维直径 (m) (介质类型为"纤维"时必填)
    K: float = None,  # 直接输入渗透率 (m²) (介质类型为"自定义"时必填)
    char_length: float = None,  # 特征长度 (m) (介质类型为"自定义"时需提供Re计算)
) -> dict:
    """
    计算低雷诺数多孔介质的压降、渗透率和雷诺数。

    返回:
        dict: 包含键 'delta_P'(压降, Pa), 'K'(渗透率, m²), 'Re'(雷诺数), 'U'(表观流速, m/s)
    """
    # 参数验证
    if U is None and (Q is None or A is None):
        raise ValueError("必须提供表观流速U或流量Q与面积A")
    if media_type == "颗粒" and dp is None:
        raise ValueError("颗粒介质需提供颗粒直径dp")
    if media_type == "纤维" and df is None:
        raise ValueError("纤维介质需提供纤维直径df")
    if media_type == "自定义" and (K is None or char_length is None):
        raise ValueError("自定义介质需提供渗透率K和特征长度char_length")

    # 计算表观流速U
    U_calc = U if U is not None else Q / A

    # 计算渗透率K
    if media_type == "颗粒":
        K_calc = (phi**3 * dp**2) / (180 * (1 - phi) ** 2)
        Re_char_length = dp
    elif media_type == "纤维":
        K_calc = (df**2 * phi**3) / (16 * (1 - phi) ** 2)
        Re_char_length = df
    else:  # 自定义介质
        K_calc = K
        Re_char_length = char_length

    # 计算雷诺数和压降
    Re = (rho * U_calc * Re_char_length) / (mu * (1 - phi))
    delta_P = (mu * L * U_calc) / K_calc

    return {
        "delta_P": delta_P,
        "K": K_calc,
        "Re": Re,
        "U": U_calc,
        "message": (
            "Re < 1, 达西定律有效" if Re < 1 else "警告: Re ≥ 1, 建议使用Brinkman修正"
        ),
    }


"""
Q_in
1.0102605128653409E-12
5.05166107609949E-12
2.0206707563861332E-11
1.0094166905231766E-10
1.0093975077968275E-9
"""

print(3.6545067255297204e-10 / 9.999999870351967e-10)


# def calculate_reynolds_number(d_n, rho, V, mu):
#     """
#     计算雷诺数(Re)基于给定的参数

#     参数:
#         d_n (float): 特征长度(管道直径) [m]
#         rho (float): 流体密度 [kg/m³]
#         V (float): 流速 [m/s]
#         mu (float): 动力粘度 [Pa·s]

#     返回:
#         float: 雷诺数
#     """
#     Re = d_n * rho * V / mu
#     return Re


# def calculate_friction_factor(Re, epsilon):
#     """
#     计算摩擦因子 f_P 基于给定的雷诺数(Re)和孔隙率(epsilon)

#     参数:
#         Re (float): 雷诺数
#         epsilon (float): 孔隙率 (0 < epsilon < 1)

#     返回:
#         float: 摩擦因子
#     """
#     numerator = 180 + 2.871 * (Re / (1 - epsilon)) ** 0.9
#     denominator = epsilon**3 * Re
#     f_p = numerator * ((1 - epsilon) ** 2) / denominator
#     return f_p


# def calculate_pressure_drop(f_p, rho, V, L, d_p):
#     """
#     计算压降 ΔP 基于摩擦因子公式

#     参数:
#         f_p (float): 摩擦因子
#         rho (float): 流体密度 [kg/m³]
#         V (float): 流速 [m/s]
#         L (float): 管道长度 [m]
#         d_p (float): 特征长度 [m]

#     返回:
#         float: 压降 [Pa]
#     """
#     delta_P = -f_p * rho * V**2 * L / d_p
#     return delta_P


# # 示例参数

# Q = 1.0093975077968275E-9  # 体积流量 [m³/s]
# A = 1e-3**2  # 横截面积 [m²]
# d_p = 1e-4  # 颗粒直径 [m]
# rho = PARAM.rho_void  # 流体密度 [kg/m³]
# V = Q / A  # 流速 [m/s]
# mu = PARAM.mu_void  # 动力粘度 [Pa·s]
# epsilon = 0.365  # 孔隙率
# L = 1e-3  # 管道长度 [m]

# # 计算流程
# Re = calculate_reynolds_number(d_p, rho, V, mu)
# f_p = calculate_friction_factor(Re, epsilon)
# delta_P = calculate_pressure_drop(f_p, rho, V, L, d_p)

# # 输出结果
# print(f"计算得到的雷诺数为: {Re:.2f}")
# print(f"计算得到的摩擦因子为: {f_p:.6f}")
# print(f"计算得到的压降为: {delta_P:.2f} Pa")


param_N_10_000 = {
    "mu": PARAM.mu_void,
    "rho": PARAM.rho_void,
    "phi": 0.365,
    "L": 1e-3,
    "Q": 1.0094166905231766e-10,
    "A": (1e-3) ** 2,
    "dp": 1e-4,
}

param_N_5_000 = {
    "mu": PARAM.mu_void,
    "rho": PARAM.rho_void,
    "media_type":"颗粒",
    "phi": 0.365,
    "L": 5e-4,
    "Q": 5.06e-10,
    "A": (5e-4) ** 2,
    "dp": 1e-4,
}
# 示例调用
if __name__ == "__main__":
    # 示例1: 颗粒介质，直接提供U
    result1 = calculate_pressure_drop(**param_N_5_000)
    print("示例1结果:", result1)

    # # 示例2: 纤维介质，提供Q和A
    # result2 = calculate_pressure_drop(
    #     mu=0.001, rho=1000, media_type="纤维", phi=0.5, L=0.2, Q=1e-6, A=1e-3, df=5e-5
    # )
    # print("示例2结果:", result2)
