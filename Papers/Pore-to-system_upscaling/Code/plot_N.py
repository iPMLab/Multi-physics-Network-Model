import numpy as np
import matplotlib
from Common_Vars import _Path_fig, plt, figsize_x, figsize_y
from scipy.optimize import curve_fit, minimize
from scipy import interpolate
from matplotlib.ticker import MultipleLocator


# 定义单调递增的二次函数（a ≥ 0）
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


# 设置约束条件确保单调性（在数据范围内导数非负）
def monotonic_constraint(params, x_data):
    a, b, c = params
    # 确保在数据范围内导数 >= 0
    x_min, x_max = np.min(x_data), np.max(x_data)
    derivative_min = 2 * a * x_min + b
    derivative_max = 2 * a * x_max + b
    return min(derivative_min, derivative_max)  # 应该 >= 0


# 数据
Ns_all = np.array((2.5, 3.455, 4.355, 4.869, 5, 10))
Ns = Ns_all[:-1]
epsilon_r = np.array((0.9, 0.942, 0.932, 0.9285, 0.932, 0.922))[:-1]
epsilon_l = np.array((1.4, 1.3, 1.25, 1.25, 1.15, 1.17))[:-1]
mean_relative_error_P = (np.array((0.1050, 0.086, 0.045, 0.029, 0.045, 0.0008)))[:-1]
mean_relative_error_hf = (np.array((0.113, 0.065, 0.04, 0.004, 0.011, 0.0006)))[:-1]

target_epsilon_r = 0.922
target_epsilon_l = 1.16


def gts(h=0, m=0, s=0):
    # get time seconds
    return int(h * 60 * 60 + m * 60 + s)


time_N2_500 = np.array(
    (
        # gts(m=20, s=21) + gts(m=8, s=5),
        # gts(m=8, s=43) + gts(m=12, s=10),
        # gts(m=16, s=3) + gts(m=12, s=6),
        gts(m=8, s=55) + gts(m=13, s=6),
        # gts(m=27, s=38) + gts(m=11, s=43),
    )
)

time_N3_455 = np.array(
    (
        # gts(h=1, m=4, s=20) + gts(h=1, m=0, s=37),
        # gts(h=0, m=36, s=5) + gts(h=1, m=14, s=17),
        # gts(h=0, m=34, s=17) + gts(h=1, m=14, s=25),
        # gts(h=0, m=32, s=14) + gts(h=0, m=54, s=29),
        gts(h=1, m=5, s=6)
        + gts(h=0, m=54, s=41),
    )
)
time_N4_353 = np.array(
    (
        # gts(h=1, m=44, s=55) + gts(h=2, m=48, s=34),
        # gts(h=1, m=16, s=43) + gts(h=3, m=23, s=41),
        # gts(h=1, m=43, s=7) + gts(h=2, m=59, s=49),
        gts(h=1, m=1, s=30) + gts(h=2, m=4, s=47),
        # gts(h=2, m=35, s=52) + gts(h=1, m=20, s=25),
    )
)

# time_N4_689 = np.array(
#     (
#         gts(h=2, m=14, s=34) + gts(h=2, m=38, s=6),
#         gts(h=1, m=36, s=7) + gts(h=2, m=40, s=36),
#         gts(h=2, m=14, s=38) + gts(h=0, m=51, s=9),
#         gts(h=1, m=24, s=42) + gts(h=2, m=22, s=34),
#         gts(h=3, m=32, s=16) + gts(h=1, m=41, s=1),
#     )
# )
time_N4_869 = np.array(
    (
        gts(h=2, m=21, s=8) + gts(h=2, m=9, s=37),
        # gts(h=1, m=48, s=11) + gts(h=2, m=13, s=44),
        # gts(h=2, m=40, s=49) + gts(h=2, m=43, s=52),
        # gts(h=1, m=39, s=32) + gts(h=1, m=56, s=26),
        # gts(h=3, m=36, s=43) + gts(h=2, m=10, s=57),
    )
)
time_N5_000 = np.array(
    (
        gts(h=2, m=28, s=31) + gts(h=3, m=29, s=6),
        # gts(h=2, m=5, s=4) + gts(h=3, m=9, s=3),
        # gts(h=2, m=39, s=33) + gts(h=2, m=32, s=16),
        # gts(h=2, m=0, s=17) + gts(h=1, m=49, s=29),
        # gts(h=3, m=51, s=59) + gts(h=1, m=10, s=53),
    )
)
time_N10_000 = np.array((gts(h=16, m=17, s=53) + gts(h=3, m=53, s=24)))

times = [time_N2_500, time_N3_455, time_N4_353, time_N4_869, time_N5_000]
for i in times:
    # print(i)
    median_index = np.argsort(i)[i.size // 2]
    print(median_index)


time_N2_500_max = time_N2_500.max()
time_N3_455_max = time_N3_455.max()
time_N4_353_max = time_N4_353.max()
time_N4_869_max = time_N4_869.max()
time_N5_000_max = time_N5_000.max()
time_N10_000_max = time_N10_000.max()

print((time_N4_353_max) / time_N10_000_max * 100)

time_N2_500_mean = time_N2_500.mean()
time_N3_455_mean = time_N3_455.mean()
time_N4_353_mean = time_N4_353.mean()
time_N4_869_mean = time_N4_869.mean()
time_N5_000_mean = time_N5_000.mean()
time_N10_000_mean = time_N10_000.mean()


# 绘制两条折线
# ax.scatter(Ns, epsilon_r, marker="o", label="$ε_r$", color="C0")
# # coefficients_r = np.polyfit(Ns, epsilon_r, 3)  # 2表示二次多项式
# # polynomial_r = np.poly1d(coefficients_r)
# # xs_r = np.linspace(Ns.min(), Ns.max(), 100)
# # fit_line_r = polynomial_r(xs_r)
# # ax.plot(xs_r, fit_line_r, linestyle="-", color="C0")


# ax.scatter(Ns, epsilon_l, marker="D", label="$ε_l$", color="C1")
# # coefficients_l = np.polyfit(Ns, epsilon_l, 3)
# # polynomial_l = np.poly1d(coefficients_l)
# # xs_l = np.linspace(Ns.min(), Ns.max(), 100)
# # fit_line_l = polynomial_l(xs_l)
# # ax.plot(xs_l, fit_line_l, linestyle="-", color="C1")

# ax.hlines(
#     target_epsilon_r,
#     xmin=Ns.min(),
#     xmax=Ns.max(),
#     linestyles="--",
#     colors="C0",
#     label="$ε_{r,target}$",
# )
# ax.hlines(
#     target_epsilon_l,
#     xmin=Ns.min(),
#     xmax=Ns.max(),
#     linestyles="--",
#     colors="C1",
#     label="$ε_{l,target}$",
# )
# ax.set_xlabel("$N$", )
# ax.set_ylabel("$ε$", )

# # 添加图例
# ax.legend()

# # 显示网格
# # ax.grid(True, linestyle="--", alpha=0.7)

# # 调整布局
# plt.tight_layout()

# # 显示图形
# plt.show()

Ns_all = np.array((2.5, 3.455, 4.355, 4.869, 5, 10))
Vs_all = np.array((1 / 64, 1 / 24, 1 / 12, 1 / 9, 1 / 8, 1 / 1))
Vs = np.array(
    (
        1 / 64,
        1 / 24,
        1 / 12,
        1 / 9,
        1 / 8,
    )
)
fig, axs = plt.subplots(1, 2, figsize=(figsize_x * 2, figsize_y))
ax1, ax2 = axs
coefficients_P = np.polyfit(Vs, mean_relative_error_P, 2)
polynomial_P = np.poly1d(coefficients_P)
xs_P = np.linspace(Vs.min(), Vs.max(), 100)
fit_line_P = polynomial_P(xs_P)
ax1.plot(xs_P, fit_line_P, linestyle="--", color="C0")
ax1.scatter(
    Vs,
    mean_relative_error_P,
    marker="o",
    label="Pressure drop",
    color="C0",
)


coefficients_hf = np.polyfit(Vs, mean_relative_error_hf, 2)
polynomial_hf = np.poly1d(coefficients_hf)
xs_hf = np.linspace(Vs.min(), Vs.max(), 100)
fit_line_hf = polynomial_hf(xs_hf)
ax1.plot(xs_hf, fit_line_hf, linestyle="--", color="C1")
ax1.scatter(Vs, mean_relative_error_hf, marker="o", label="Heat transfer", color="C1")
# ax1.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5)
ax1.scatter(Vs[2], 0, marker="o", color="C3", zorder=100, clip_on=False)

ax1.plot(
    [Vs[2], Vs[2]],
    [0, mean_relative_error_P[2]],
    c="C3",
    linewidth=1.5,
    linestyle="--",
)
ax1.scatter(
    Vs[2], mean_relative_error_P[2], marker="o", color="C3", zorder=100, clip_on=False
)
# ax1.text(Vs[2] + 0.0125, 0.003, f"{Vs[2]:.3f}", ha="right", va="center", color="C3")

ax1.scatter(
    0, mean_relative_error_P[2], marker="o", color="C3", zorder=100, clip_on=False
)
# ax1.text(
#     Vs.min() - 0.029,
#     mean_relative_error_P[2],
#     f"{mean_relative_error_P[2]:.3f}",
#     color="C3",
#     va="center",
# )
ax1.plot(
    [Vs[2], 0],
    [mean_relative_error_P[2], mean_relative_error_P[2]],
    c="C3",
    linewidth=1.5,
    linestyle="--",
)
ax1.set_xlabel("ROl volume (relative to full-size system)\n(a)")
ax1.set_ylabel("Relative error")
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# 添加图例
ax1.legend(frameon=False, loc="upper right")


# times = np.array(
#     (
#         time_N2_500_mean,
#         time_N3_455_mean,
#         time_N4_689_mean,
#         time_N4_869_mean,
#         time_N5_000_mean,
#         time_N10_000_mean,
#     )
# )

times_export = np.array(
    (
        77.97,
        235.28,
        570.63,
        920,
        1023.82,
    )
)


times_extract = np.array(
    (
        19,
        25,
        26,
        27,
        28,
    )
)


times_tuning = np.array(
    (
        0.68,
        1.1,
        1.15,
        1.27,
        1.3,
    )
)

times_dnm = (times_export + times_extract + times_tuning * 20) / 60 / 60

times_dns = (
    np.array(
        (
            time_N2_500_max,
            time_N3_455_max,
            time_N4_353_max,
            time_N4_869_max,
            time_N5_000_max,
            # time_N10_000_max,
        )
    )
    / 60
    / 60
)


times = times_dns + times_dnm


Vs_all = Vs


plt.axhline(
    y=time_N10_000_max / 60 / 60,
    color="C0",
    linestyle="--",
    linewidth=1.5,
    label="DNS for full-size system",
)


plt.scatter(
    Vs_all,
    times_dns + times_dnm,
    marker="o",
    label="Calibrated DNM for full-size system",
    color="C1",
    alpha=1,
)


# plt.bar(Vs_all, times_dnm, bottom=times_dns, width=0.01, label="DNM")


coefficients_time = np.polyfit(Vs_all, times, 2)
polynomial_time = np.poly1d(coefficients_time)
xs_time = np.linspace(0, Vs_all.max(), 100)

# 定义损失函数
# def loss(params):
#     a, b, c = params
#     pred = quadratic(Vs_all, a, b, c)
#     return np.sum((pred - times) ** 2)


# 约束：对称轴在最大值右侧
# constraint = {
#     "type": "ineq",
#     "fun": lambda params: -params[1] / (2 * params[0]) - Vs_all.max(),
# }
# bounds = [(-np.inf, 0), (None, None), (None, None)]  # a < 0

# result = minimize(loss, initial_guess, constraints=constraint, bounds=bounds)
# coefficients_time = result.x

# 初始猜测（使用普通拟合结果）
coefficients_time = np.polyfit(Vs_all, times, 2)

# 优化（确保a < 0）
#

polynomial_time = np.poly1d(coefficients_time)
fit_line_time = polynomial_time(xs_time)
ax2.plot(xs_time, fit_line_time, linestyle="--", color="C1")
# ax2.scatter(Vs_all, times, marker="o", color="C0")

# ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.plot([Vs[2], Vs[2]], [0, times[2]], c="C3", linewidth=1.5, linestyle="--")
# ax2.text(Vs[2] + 0.105, 0.55, f"{Vs[2]:.3f}", ha="right", va="center", color="C3")
ax2.scatter(Vs[2], 0, marker="o", color="C3", zorder=100, clip_on=False)
ax2.scatter(0, times[2], marker="o", color="C3", zorder=100, clip_on=False)
# ax2.text(-0.013, times[2], f"{times[2]:.1f}", ha="right", va="center", color="C3")
ax2.plot([Vs[2], 0], [times[2], times[2]], c="C3", linewidth=1.5, linestyle="--")
ax2.scatter(Vs[2], times[2], marker="o", color="C3", zorder=100, clip_on=False)
ax2.set_xlabel("ROl volume (relative to full-size system)\n(b)")
ax2.set_ylabel("Total computational time (hours)")
ax2.set_ylim(bottom=0, top=27)
ax2.set_xlim(left=0)
ax2.legend(frameon=False, loc="upper left")
# ax.grid(True, linestyle="--", alpha=0.7)


plt.tight_layout()
ax2.legend(frameon=False, edgecolor="none", loc="upper left")
plt.savefig(_Path_fig / "N.png")
plt.show()
