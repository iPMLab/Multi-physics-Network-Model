import numpy as np
import pyvista as pv
import pandas as pd
from pathlib import Path
import numpy as np


Path_mesh = Path(r"C:\Users\yjp\Desktop\test\mesh_o.vtu")
mesh = pv.read(Path_mesh, progress_bar=True)
mesh.cell_data["Material"] = np.ones(mesh.n_cells)


points = mesh.points

cells = mesh.cells.reshape(-1, mesh.cells[0] + 1)[:, 1:5]
node_ids = cells + 1  # 所有节点的索引+1

with open(r"C:\Users\yjp\Desktop\test\output.bdf", "w") as f:
    # GRID 行
    np.savetxt(
        f,
        np.column_stack([np.arange(1, len(points) + 1), points]),
        fmt="GRID,%d,,%.6f,%.6f,%.6f",
        delimiter="",
    )
    # CTETRA 行
    np.savetxt(
        f,
        np.column_stack(
            [np.arange(1, len(cells) + 1), mesh.cell_data["Material"], node_ids]
        ),
        fmt="CTETRA,%d,%d,%d,%d,%d,%d",
        delimiter="",
    )
