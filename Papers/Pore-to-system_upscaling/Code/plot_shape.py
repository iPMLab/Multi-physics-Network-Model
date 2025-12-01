import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建图像
fig, ax = plt.subplots()

# 定义箭头参数
x_tail = 0.2
y_tail = 0.5
x_head = 0.8
y_head = 0.5

# 使用FancyArrowPatch创建一个自定义样式的箭头
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                  arrowstyle=mpatches.ArrowStyle.Fancy(
                                      head_length=.45,
                                      head_width=.35,
                                      tail_width=.1),  # 调整这些参数以获得鱼尾效果
                                  mutation_scale=20,  # 影响箭头大小
                                  linewidth=1,
                                  color='blue')

# 添加箭头到图像
ax.add_patch(arrow)

# 设置x、y轴的显示范围
plt.xlim(0, 1)
plt.ylim(0.2, 0.8)

# 隐藏坐标轴
plt.axis('off')

# 显示图像
plt.show()