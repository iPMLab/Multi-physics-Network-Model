import mph
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
path_comsol_mph = r"D:\yjp\OneDrive - zju.edu.cn\Code\ZJU\Study\Python\heat_tranfer\3D_Study\COMSOL\big\3D_Finney_results\Finney.mph"
Path_comsol_mph = Path(path_comsol_mph)
client = mph.start()
model = client.load(Path_comsol_mph)
model_java = model.java
geometry_node=model/'geometries/Geometry 1'
geometry_node.children()
sphere_nodes=[i for i in geometry_node.children() if 'Sphere' in str(i) and i.java.isActive()]
# spheres=[i for i in geometry_nodes_name if 'Sphere' in i]

sphere_properties=np.zeros((len(sphere_nodes),4))
for i,sphere_node in tqdm(enumerate(sphere_nodes)):
    sphere_properties[i,0]=sphere_node.property('x')
    sphere_properties[i,1]=sphere_node.property('y')
    sphere_properties[i,2]=sphere_node.property('z')
    sphere_properties[i,3]=sphere_node.property('r')

sphere_properties=pd.DataFrame(data=sphere_properties,columns=['x','y','z','r'],index=None)
sphere_properties.to_csv(Path_comsol_mph.with_suffix('.csv'),index=False)