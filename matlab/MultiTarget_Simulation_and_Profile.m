% =========================================================================
% task_multi_analysis.m —— 多点目标仿真 & 单点主瓣宽度定量分析
% 说明：本脚本独立运行，不修改你上周的 simulate_point.m
% =========================================================================
clear; clc; close all;

%% ===================== 1. SAR 系统参数 =====================
C       = 299792458;                 
fc      = 5.405e9;                   
lambda  = C / fc;                    
Fr      = 8.341e7;                   
PRF     = 1612.8;                    
Vr      = 7100;                      
R0      = 8.5e5;                     
Tp      = 50e-6;                     
Br      = 40e6;                      
Kr      = Br / Tp;                   
Fdc     = 0;                         
Na      = 1024;                      
Nr      = 2048;                      

ta   = (-Na/2 : Na/2-1) / PRF;       
tr   = (-Nr/2 : Nr/2-1) / Fr + 2*R0/C; 

% 将上周的 RD 算法封装成一个内联函数，方便重复调用（仿真和分析各用一次）
RD_Process = @(echo) run_rd_algorithm(echo, Na, Nr, Fr, PRF, Vr, R0, Tp, Br, Kr, Fdc, lambda);

%% ===================== 任务一：多点目标仿真 =====================
fprintf('======== 开始多点目标仿真 ========\n');
% 设置 5 个目标 [方位向偏移(m), 距离向偏移(m), 反射系数]
% 间距拉大 (方位 ±300m, 距离 ±200m), 使各点主瓣互不干扰, 旁瓣可见
targets = [
      0,      0,    1.0;   % 中心
   -300,    150,    0.8;   % 左上
    250,   -100,    0.9;   % 右下
   -200,   -200,    0.7;   % 左下
    300,    180,    0.6    % 右上
];

echo_multi = zeros(Nr, Na);
for k = 1:size(targets, 1)
    dx_az = targets(k, 1); 
    dR_rg = targets(k, 2); 
    sigma = targets(k, 3);
    
    % 目标的瞬时斜距
    Rn = sqrt((R0 + dR_rg)^2 + (Vr * ta - dx_az).^2);
    
    for ii = 1 : Na
        tau = tr - 2 * Rn(ii) / C;                 
        rectWin = double(abs(tau) <= Tp / 2);     
        echo_multi(:, ii) = echo_multi(:, ii) + ...
            sigma * (rectWin .* exp(1j * pi * Kr * tau.^2) ...
            .* exp(-1j * 4 * pi * Rn(ii) / lambda)).';
    end
end

% 运行 RD 算法成像
DATA_multi = RD_Process(echo_multi);
img_multi = abs(DATA_multi);
imgNorm_multi = img_multi / max(img_multi(:));

% 绘制多点图像并保存
figure('Name','多点目标 RD 成像结果','Color','w');
imagesc(db(imgNorm_multi + 1e-10), [-40, 0]); 
colormap(gray); axis image; colorbar;
xlabel('方位向'); ylabel('距离向'); title('多点目标 RD 成像结果 (仿真基线)');
saveas(gcf, 'sim_point_multi.jpg');
fprintf('✅ 多点成像完成，已保存 sim_point_multi.jpg\n');

%% ===================== 任务二：定量分析 (单点理想情况) =====================
fprintf('\n======== 开始单点定量分析 ========\n');
% 生成单点回波用于测量主瓣宽度
% 加 0.5 像素的亚像素偏移, 避免峰值正好落在整数采样点上 (减小离散采样伪影)
dR = C / (2 * Fr);
dA = Vr / PRF;
offset_R = 0.5 * dR;     % 距离向 0.5 像素偏移
offset_A = 0.5 * dA;     % 方位向 0.5 像素偏移
echo_single = zeros(Nr, Na);
for ii = 1 : Na
    Rn_center = sqrt((R0 + offset_R)^2 + (Vr * ta(ii) - offset_A)^2);
    tau = tr - 2 * Rn_center / C;
    rectWin = double(abs(tau) <= Tp / 2);
    echo_single(:, ii) = (rectWin .* exp(1j * pi * Kr * tau.^2) ...
                         .* exp(-1j * 4 * pi * Rn_center / lambda)).';
end

DATA_single = RD_Process(echo_single);
img_single = abs(DATA_single);
imgNorm_single = img_single / max(img_single(:));

% 寻找峰值
[~, idxR] = max(max(img_single, [], 2));
[~, idxC] = max(max(img_single, [], 1));

% 定位误差 (相对于图像中心, 像素)
centerR = Nr / 2 + 0.5;
centerC = Na / 2 + 0.5;
loc_err = sqrt((idxR - centerR)^2 + (idxC - centerC)^2);

% 提取剖面与 3dB 阈值
profile_R = imgNorm_single(:, idxC);
profile_A = imgNorm_single(idxR, :);
threshold = 1 / sqrt(2);

% --- 距离向 -3dB 宽度 (插值) ---
idxR_l_int = find(profile_R(1:idxR) < threshold, 1, 'last');
idxR_r_int = find(profile_R(idxR:end) < threshold, 1, 'first') + idxR - 1;
idxR_l = interp1(profile_R(idxR_l_int:idxR_l_int+1), [idxR_l_int, idxR_l_int+1], threshold);
idxR_r = interp1(profile_R(idxR_r_int-1:idxR_r_int), [idxR_r_int-1, idxR_r_int], threshold);
width_R_pixel = idxR_r - idxR_l;

% --- 方位向 -3dB 宽度 (插值) ---
idxA_l_int = find(profile_A(1:idxC) < threshold, 1, 'last');
idxA_r_int = find(profile_A(idxC:end) < threshold, 1, 'first') + idxC - 1;
idxA_l = interp1(profile_A(idxA_l_int:idxA_l_int+1), [idxA_l_int, idxA_l_int+1], threshold);
idxA_r = interp1(profile_A(idxA_r_int-1:idxA_r_int), [idxA_r_int-1, idxA_r_int], threshold);
width_A_pixel = idxA_r - idxA_l;

% --- 理论分辨率对比 ---
resR_measured = width_R_pixel * dR;
resA_measured = width_A_pixel * dA;
Ba_theory = 2 * Vr^2 * (Na/PRF) / (lambda * R0);
resR_theory = C / (2 * Br);
resA_theory = Vr / Ba_theory;

fprintf('\n================ 定量分析结果 ================\n');
fprintf('峰值坐标: 行 %d, 列 %d (中心: %.1f, %.1f)\n', idxR, idxC, centerR, centerC);
if loc_err < 1
    loc_status = '满足';
else
    loc_status = '不满足';
end
fprintf('定位误差: %.2f 像素 (要求 < 1 像素: %s)\n', loc_err, loc_status);
fprintf('【距离向】\n  实测 -3dB 宽度: %.2f 像素 (物理宽度: %.2f m)\n', width_R_pixel, resR_measured);
fprintf('  理论分辨率: %.2f m, 比值: %.2f\n', resR_theory, resR_measured/resR_theory);
fprintf('【方位向】\n  实测 -3dB 宽度: %.2f 像素 (物理宽度: %.2f m)\n', width_A_pixel, resA_measured);
fprintf('  理论分辨率: %.2f m, 比值: %.2f\n', resA_theory, resA_measured/resA_theory);
fprintf('==============================================\n');

% --- 绘制剖面图 ---
figure('Name','主瓣宽度定量分析','Color','w');
subplot(2,1,1);
plot(profile_R, 'b', 'LineWidth', 1.5); hold on;
yline(threshold, 'r--', 'Label', '-3dB');
xline(idxR_l, 'k--', 'LineWidth', 1.2); xline(idxR_r, 'k--', 'LineWidth', 1.2);
xlim([idxR-50, idxR+50]); grid on;
title('距离向剖面 (Range Profile)'); xlabel('像素'); ylabel('归一化幅度');

subplot(2,1,2);
plot(profile_A, 'b', 'LineWidth', 1.5); hold on;
yline(threshold, 'r--', 'Label', '-3dB');
xline(idxA_l, 'k--', 'LineWidth', 1.2); xline(idxA_r, 'k--', 'LineWidth', 1.2);
xlim([idxC-50, idxC+50]); grid on;
title('方位向剖面 (Azimuth Profile)'); xlabel('像素'); ylabel('归一化幅度');

saveas(gcf, 'profile_3dB.jpg');
fprintf('✅ 剖面图已保存 profile_3dB.jpg\n');


%% ===================== 辅助函数：RD 成像算法 =====================
function DATA = run_rd_algorithm(echo, Na, Nr, Fr, PRF, Vr, R0, Tp, Br, Kr, Fdc, lambda)
    % 这段直接复用了你上周 simulate_point.m 中的核心算法
    DATA = echo; 
    Np = ceil(Tp * Fr);
    tnrn = (-Nr/2 : Nr/2-1) / Fr;
    fr = (-Nr/2 : Nr/2-1) / Nr * Fr;
    fa = (-Na/2 : Na/2-1) / Na * PRF;
    SquintAngle = asin(lambda * Fdc / 2 / Vr);
    Rs = R0;
    dR = 299792458 / (2 * Fr);
    Rref = Rs * cos(SquintAngle);

    % 3.1 距离压缩
    NfftR = 2^nextpow2(Np + Nr - 1);
    NHf = (-NfftR/2 : NfftR/2-1).' / NfftR * Fr;
    Nzero = fix((Fr - Br) / Fr * NfftR / 2) * 2;
    WinR = padarray(ones(NfftR - Nzero, 1), [Nzero/2 0], 0);
    Hf = WinR .* exp(1j * pi * NHf.^2 ./ Kr); 
    temp = DATA .* exp(-1j * 2 * pi * Fdc * ta_gen(Na, PRF));
    temp = padarray(temp, [(NfftR - Nr)/2 0], 0);
    temp = fftshift(ifft(fft(ifftshift(temp, 1)) .* ifftshift(Hf, 1)), 1);
    DATA = temp(NfftR/2 - Nr/2 + 1 : NfftR/2 + Nr/2, :);

    % 3.2 SRC
    fa = fa + Fdc;
    DATA = fftshift(fft2(ifftshift(DATA)));
    D = sqrt(1 - (lambda * fa / 2 / Vr).^2);
    Ksrc = 2 * Vr^2 * 5.405e9^3 * D.^3 ./ (299792458 * Rref * fa.^2);
    Ksrc(fa == 0) = inf;
    fr = fr.';
    Hsrc = exp(-1j * pi * fr.^2 ./ Ksrc);
    DATA = DATA .* Hsrc;
    DATA = fftshift(ifft(ifftshift(DATA, 1), [], 1), 1);

    % 3.3 RCMC
    Rgrid = Rs + tnrn * 299792458 / 2;
    RCM = Rgrid.' ./ D - Rgrid.';
    N_RCM = ceil(RCM / dR);
    F_RCM = N_RCM - RCM / dR;
    NKel = 8;
    DATA_RCMC = zeros(Nr, Na);
    for aa = 1 : Na
        for rr = 1 : Nr
            SincKel = sinc(F_RCM(rr,aa) - NKel/2 : 1 : F_RCM(rr,aa) + NKel/2 - 1);
            SincKel = SincKel / sum(SincKel);
            kk = rr + N_RCM(rr, aa);
            kk = rem(kk, Nr);
            if kk == 0, kk = Nr; end
            if (kk - NKel/2) < 1 || (kk + NKel/2 - 1) > Nr
                nn = mod((kk - NKel/2 : kk + NKel/2 - 1) - 1, Nr) + 1;
            else
                nn = (kk - NKel/2) : 1 : (kk + NKel/2 - 1);
            end
            DATA_RCMC(rr, aa) = SincKel * DATA(nn, aa);
        end
    end
    DATA = DATA_RCMC; 

    % 3.4 方位压缩
    NfftA = 2^nextpow2(2 * Na);
    Fnan = Fdc + (-NfftA/2 : NfftA/2-1) / NfftA * PRF;
    fbase = (-NfftA/2 : NfftA/2-1) / NfftA * PRF;
    D = sqrt(1 - (lambda * Fnan / 2 / Vr).^2);
    Dc = sqrt(1 - (lambda * Fdc / (2 * Vr))^2);
    Ha = exp(1j * 4 * pi * Rgrid.' * D * 5.405e9 / 299792458);
    eta_c = -lambda * Rgrid.' * Fdc / (2 * Vr^2 * Dc);
    Ha = Ha .* exp(-1j * 2 * pi * eta_c * fbase);
    temp = fftshift(ifft(ifftshift(DATA, 2), [], 2), 2);
    temp = padarray(temp, [0 (NfftA - Na)/2], 0);
    for rr = 1 : Nr
        temp(rr, :) = fftshift(ifft(fft(ifftshift(temp(rr, :))) .* ifftshift(Ha(rr, :))));
    end
    DATA = temp(:, NfftA/2 - Na/2 + 1 : NfftA/2 + Na/2);
end

function ta = ta_gen(Na, PRF)
    ta = (-Na/2 : Na/2-1) / PRF;
end