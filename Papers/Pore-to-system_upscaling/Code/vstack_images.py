import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Common_Vars import _Path_fig

# 加载两张图片
img1 = mpimg.imread(_Path_fig / "get_data_single_N5.000.png")
img2 = mpimg.imread(_Path_fig / "small_pore_distribution_N5.000.png")

# 创建画布和子图（1行2列）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 4.8 * 2))

# 显示图片并关闭坐标轴
ax1.imshow(img1)
ax1.axis("off")  # 关闭坐标轴
ax2.imshow(img2)
ax2.axis("off")
plt.subplots_adjust(wspace=0, hspace=-0.25)
# 自动调整间距并保存或显示
plt.tight_layout()
plt.savefig(
    _Path_fig / "combined_N5.000.png",
    dpi=330,
    bbox_inches=None,
    pad_inches=0,
)  # 保存合并后的图
plt.show()
