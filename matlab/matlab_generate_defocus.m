% 读取干净图像 (使用你已经 recover 出来的 ROI 文件)
load('C:\Users\d\Desktop\SAR_Project - 副本\data\ROIs\ROI_1_complex.mat'); % 包含 roi_complex
img_clean = double(roi_complex);

% 方位向 = 列 = 最后一维（对应 Python 里的 axis=1）
Naz = size(img_clean, 2);

% 设定相位误差系数 (二次主导，对应 Q = alpha2*pi^2, 例如 Q=10 rad -> alpha2 = 10/pi^2)
alpha2 = 10 / (pi^2);
alpha3 = 0.1 * alpha2;

% 构造频率轴 (fftfreq 的中心化等价，顺序与 fftshift 输出一致)
u = (-Naz/2 : Naz/2-1) / Naz;
w = 2 * pi * u;

% 计算相位向量 phi (1×Naz 行向量，沿方位向 dim2 隐式广播)
phi = alpha2 * (w.^2) + alpha3 * (w.^3);

% 频域注入：先沿方位向(dim2)做中心化 FFT，乘 exp(+j*phi)，再 IFFT 回来
Img_F = fftshift(fft(ifftshift(img_clean, 2), [], 2), 2);
Img_F_defocus = Img_F .* exp(1i * phi);
img_defocus = fftshift(ifft(ifftshift(Img_F_defocus, 2), [], 2), 2);

% 保存为输入散焦图 (与 matlab_min_entropy_autofocus.m 的加载路径一致)
% 同时存下真实 alpha 系数，供自聚焦脚本核对收敛精度
input_defocus = single(img_defocus);
save('C:\Users\d\Desktop\SAR_Project - 副本\data\matlab_defocus_sample.mat', ...
     'input_defocus', 'alpha2', 'alpha3');
disp('散焦样本已生成！保存至 data/matlab_defocus_sample.mat');
