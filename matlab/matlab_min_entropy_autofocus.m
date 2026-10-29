% ===== 最小熵自聚焦主脚本 =====
clc; clear;

% 1. 加载散焦样本（二选一，改 INPUT_FILE）
%    (A) 对比 RAFnet：输入由 compare_baselines.py 导出到 data\matlab_input_raw.mat
%    (B) 独立基线验证：由 matlab_generate_defocus.m 生成 data\matlab_defocus_sample.mat
INPUT_FILE = 'C:\Users\d\Desktop\SAR_Project - 副本\data\matlab_input_raw.mat';        % (A) 对比 RAFnet
% INPUT_FILE = 'C:\Users\d\Desktop\SAR_Project - 副本\data\matlab_defocus_sample.mat'; % (B) 独立基线验证
load(INPUT_FILE);   % 变量 input_defocus
Naz = size(input_defocus, 2);   % 方位向 = 列 = 最后一维
Nrg = size(input_defocus, 1);   % 距离向 = 行
u = (-Naz/2 : Naz/2-1) / Naz;
w = 2 * pi * u;

% 若生成脚本存了真实系数，打印出来用于核对
if exist('alpha2', 'var') && exist('alpha3', 'var')
    fprintf('真实注入：alpha2=%.4f, alpha3=%.4f (Q=%.2f rad)\n', alpha2, alpha3, alpha2*pi^2);
end

% 2. 目标函数 (per-range-bin 能量加权熵，见文件末尾局部函数)
obj_fun = @(alpha) compute_entropy(input_defocus, alpha, w);

% 3. 先固定 alpha3=0，在 alpha2 上做一维粗网格搜索 (Q = alpha2*pi^2，扫 Q∈[5,25] rad)
a2_grid = linspace(0.5, 2.5, 41);
E_grid = arrayfun(@(a) obj_fun([a, 0]), a2_grid);
[~, k] = min(E_grid);
a2_init = a2_grid(k);
fprintf('网格粗扫：alpha2 ≈ %.4f (Q ≈ %.2f rad)\n', a2_init, a2_init*pi^2);

% 4. 从网格最优处出发，用 fminsearch 精修 [alpha2, alpha3]
options = optimset('Display', 'iter', 'MaxIter', 300, 'TolX', 1e-6, 'TolFun', 1e-8);
disp('开始最小熵优化...');
alpha_opt = fminsearch(obj_fun, [a2_init, 0], options);

% 5. 使用优化出的参数进行补偿
phi = alpha_opt(1) * (w.^2) + alpha_opt(2) * (w.^3);
Img_F = fftshift(fft(ifftshift(input_defocus, 2), [], 2), 2);
Img_F_comp = Img_F .* exp(-1i * phi);
img_comp_matlab = fftshift(ifft(ifftshift(Img_F_comp, 2), [], 2), 2);

% 6. 保存结果供 Python 对比调用 (变量名 img_comp_matlab 与 compare_baselines.py 一致)
save('C:\Users\d\Desktop\SAR_Project - 副本\data\matlab_autofocus_result.mat', ...
     'img_comp_matlab', 'alpha_opt', 'input_defocus', 'a2_grid', 'E_grid');
fprintf('优化完成：alpha2=%.4f, alpha3=%.4f (Q=%.2f rad)\n', ...
        alpha_opt(1), alpha_opt(2), alpha_opt(1)*pi^2);
disp('结果已保存至 data/matlab_autofocus_result.mat');

% ===== 局部函数：per-range-bin 能量加权熵 =====
function ent = compute_entropy(img, alpha, w)
    phi = alpha(1) * (w.^2) + alpha(2) * (w.^3);
    Img_F = fftshift(fft(ifftshift(img, 2), [], 2), 2);
    Img_F_comp = Img_F .* exp(-1i * phi);
    img_comp = fftshift(ifft(ifftshift(Img_F_comp, 2), [], 2), 2);

    amp = abs(img_comp);                   % Nrg × Naz
    bin_energy = sum(amp, 2);              % 每个距离 bin 的能量 (Nrg×1)
    total_energy = sum(bin_energy) + eps;  % 总能量
    p = amp ./ (bin_energy + 1e-12);       % 每个距离 bin 内归一化为分布
    H_bin = -sum(p .* log2(p + 1e-12), 2); % 每个 bin 的熵 (Nrg×1)
    ent = sum((bin_energy / total_energy) .* H_bin);  % 能量加权平均
end
