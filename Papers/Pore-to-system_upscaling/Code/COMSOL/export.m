import com.comsol.model.*
import com.comsol.model.util.*
heat_flux = 100000;
% 加载模型文件
%"D:\\yjp\\Workdir\\Code\\ZJU\\Study\\Python\\heat_tranfer\\3D_Study\\COMSOL\\big\\3D_Finney_results\\Finney_6000W_sweep_%dW.mph",heat_flux
model_name = sprintf("C:\\Users\\yjp\\Desktop\\test1.mph");
model = mphopen(model_name);

% 配置导出参数
exportConfig = model.result.export('data1');
exportConfig.set('innerinput', 'manual');
%0.1,1,10,20,50,100,200

% 定义参数序列
u_inlets = [0.00028,0.00084,0.00169,0.00281,0.02809,0.05617,0.14043];
indices = 1:length(u_inlets); % 生成连续索引数组

% 批量导出循环
for i = 1:length(u_inlets)
    % 动态生成文件名（MATLAB需用sprintf代替f-string）
    filename = sprintf('D:\\yjp\\Workdir\\Code\\ZJU\\Study\\Python\\multi-physic-network-model\\sample_data_1_1\\Finney\\comsol_data\\1_1_u%.5f_hf%dW.txt', u_inlets(i),heat_flux);
    
    % 设置导出参数
    exportConfig.set('filename', filename);
    exportConfig.set('solnum', indices(i));
    
    % 执行导出操作[8,11](@ref)
    exportConfig.run();
    
    % 进度反馈
    fprintf('已导出: %s\n', filename);
end