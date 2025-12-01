import matplotlib.pyplot as plt

# 数据
N = [2.545, 3.455, 4.353, 4.869, 5]
y = [1.2, 1.15, 1.09, 1.05, 1.05]

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制数据点
plt.plot(N, y, "bo-", markersize=8)

# 绘制理论值线
plt.axhline(y=1.05, color="r", linestyle="--", label="theoretical value = 1.05")
min_rev_N = 4.869
min_rev_epsilon = 1.05  # 对应 y 值
plt.scatter(min_rev_N, min_rev_epsilon, color="green", s=100, zorder=5)  # 高亮显示该点
plt.annotate(
    "N of REV",  # 标注文本
    xy=(min_rev_N, min_rev_epsilon),  # 标注点坐标
    xytext=(min_rev_N-0.01, min_rev_epsilon + 0.02),  # 文本位置偏移
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black"),  # 箭头样式
)


# 添加标签和标题
plt.xlabel("N", fontsize=12)
plt.ylabel(r"$\epsilon$", fontsize=12)
plt.title("N of REV", fontsize=14)

# 添加图例
plt.legend(fontsize=10)

# 设置网格
plt.grid(True, linestyle="--", alpha=0.6)

# 显示图形
plt.show()
