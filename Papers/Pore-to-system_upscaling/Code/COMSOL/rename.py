from pathlib import Path

Path_root = Path(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\heat_tranfer\3D_Study\COMSOL\big\3D_Finney_results"
)

for file in Path_root.rglob("*_alpha*"):
    if file.is_file():  # 确保是文件
        new_name = file.name.replace("_alpha", "_N")
        new_path = file.with_name(new_name)
        file.rename(new_path)
        print(f"Renamed: {file} -> {new_path}")
