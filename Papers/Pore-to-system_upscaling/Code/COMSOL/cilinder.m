import com.comsol.model.*
import com.comsol.model.util.*

model = mphopen("C:\\Users\\yjp\Desktop\\cylinder.mph");


model.geom('geom1').feature('cyl1').set('pos', [0;0;0]);
model.geom('geom1').feature('cyl1').set('axistype', 'cartesian');
model.geom('geom1').feature('cyl1').set('axis', [0.2;0.3;1.0]);
model.save("C:\\Users\\yjp\Desktop\\cylinder_test.mph")
x = 20;
