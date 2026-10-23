% =========================================================================
% simulate_point.m  ——  SAR 条带模式点目标回波仿真 + RD 成像验证
% -------------------------------------------------------------------------
% 目的: 用数学模型生成一个点目标的雷达回波, 再用 Read_Data_RD.m 中的
%       Range-Doppler 核心算法(距离压缩→SRC→RCMC→方位压缩)处理,
%       若图像中心聚焦出一个锐利亮点, 则说明 RD 算法正确有效。
% =========================================================================
clear; clc; close all;

%% ===================== 1. SAR 系统参数 =====================
C       = 299792458;                 % 光速 (m/s)
fc      = 5.405e9;                   % 载频 (Hz), 与 Sentinel-1 C 波段一致
lambda  = C / fc;                    % 波长 (m)
Fr      = 8.341e7;                   % 距离采样率 (Hz)
PRF     = 1612.8;                    % 脉冲重复频率 (Hz)
Vr      = 7100;                      % 等效雷达速度 (m/s)
R0      = 8.5e5;                     % 场景中心斜距 (m)
Tp      = 50e-6;                     % 发射脉冲宽度 (s)
Br      = 40e6;                      % 信号带宽 (Hz)
Kr      = Br / Tp;                   % 调频率 (Hz/s), 正扫频
Fdc     = 0;                         % 多普勒中心频率 (Hz), 无斜视

% 成像尺寸 (2 的幂, 利于 FFT; 取较小值以加快 RCMC 双重循环)
Na      = 1024;                      % 方位向脉冲数
Nr      = 2048;                      % 距离向采样点数

%% ===================== 2. 生成点目标回波 =====================
ta   = (-Na/2 : Na/2-1) / PRF;               % 方位时间 (s), 行向量 1xNa
tr   = (-Nr/2 : Nr/2-1) / Fr + 2*R0/C;        % 距离快时间 (s), 以目标时延为中心
R    = sqrt(R0^2 + (Vr * ta).^2);              % 瞬时斜距 (m), 1xNa

fprintf('生成点目标回波...\n');
echo = zeros(Nr, Na);
for ii = 1 : Na
    tau = tr - 2 * R(ii) / C;                 % 相对于目标时延的快时间
    rectWin = double(abs(tau) <= Tp / 2);     % 矩形窗 (脉内)
    echo(:, ii) = (rectWin .* exp(1j * pi * Kr * tau.^2) ...
                         .* exp(-1j * 4 * pi * R(ii) / lambda)).';
end
fprintf('回波尺寸: %d x %d\n', Nr, Na);

% 显示原始回波幅度
figure('Name','原始回波幅度','Color','w');
imagesc(abs(echo)); axis image; colormap(gray);
xlabel('方位向'); ylabel('距离向'); title('点目标原始回波 (含距离徙动轨迹)');

%% ===================== 3. RD 成像算法 =====================
% 以下算法结构与 Read_Data_RD.m 完全一致, 便于验证
% -------------------------------------------------------------------------
DATA      = echo;
clear echo;

Np        = ceil(Tp * Fr);                      % 脉冲采样点数
tnrn      = (-Nr/2 : Nr/2-1) / Fr;             % 相对于距离中心的快时间, 1xNr
fr        = (-Nr/2 : Nr/2-1) / Nr * Fr;        % 距离频率, 1xNr
fa        = (-Na/2 : Na/2-1) / Na * PRF;       % 方位频率 (多普勒域), 1xNa
SquintAngle = asin(lambda * Fdc / 2 / Vr);      % 斜视角
Rs        = R0;                                  % 中心斜距
dR        = C / (2 * Fr);                        % 距离采样间隔 (m)
Rref      = Rs * cos(SquintAngle);               % 参考斜距

% --- 3.1 距离压缩 ---
fprintf('距离压缩...\n');
NfftR   = 2^nextpow2(Np + Nr - 1);
NHf     = (-NfftR/2 : NfftR/2-1).' / NfftR * Fr;
Nzero   = fix((Fr - Br) / Fr * NfftR / 2) * 2;
WinR    = padarray(ones(NfftR - Nzero, 1), [Nzero/2 0], 0);
Hf      = WinR .* exp(1j * pi * NHf.^2 ./ Kr);  % 距离匹配滤波器

temp    = DATA .* exp(-1j * 2 * pi * Fdc * ta);
temp    = padarray(temp, [(NfftR - Nr)/2 0], 0);
temp    = fftshift(ifft(fft(ifftshift(temp, 1)) .* ifftshift(Hf, 1)), 1);
DATA    = temp(NfftR/2 - Nr/2 + 1 : NfftR/2 + Nr/2, :);
clear temp Hf WinR;

% --- 3.2 二次距离压缩 (SRC) ---
fprintf('二次距离压缩 (SRC)...\n');
fa      = fa + Fdc;
DATA    = fftshift(fft2(ifftshift(DATA)));
D       = sqrt(1 - (lambda * fa / 2 / Vr).^2);          % 徙动因子, 1xNa
Ksrc    = 2 * Vr^2 * fc^3 * D.^3 ./ (C * Rref * fa.^2);  % SRC 调频率, 1xNa
Ksrc(fa == 0) = inf;                                     % 避免 fa=0 奇异
fr      = fr.';                                           % 转列向量, 使 Hsrc 为 NrxNa
Hsrc    = exp(-1j * pi * fr.^2 ./ Ksrc);                 % (Nrx1)./(1xNa) -> NrxNa
DATA    = DATA .* Hsrc;
DATA    = fftshift(ifft(ifftshift(DATA, 1), [], 1), 1);
clear Hsrc Ksrc;

% --- 3.3 距离徙动校正 (RCMC, sinc 插值) ---
fprintf('距离徙动校正 (RCMC)...\n');
Rgrid   = Rs + tnrn * C / 2;
Rrefs   = Rgrid;
RCM     = Rrefs.' ./ D - Rgrid.';                         % 徙动量
N_RCM   = ceil(RCM / dR);                                  % 整数徙动单元
F_RCM   = N_RCM - RCM / dR;                                % 小数徙动单元
NKel    = 8;                                               % sinc 插值核点数

DATA_RCMC = zeros(Nr, Na);
for aa = 1 : Na
    if mod(aa, 256) == 0
        fprintf('  RCMC: %d / %d\n', aa, Na);
    end
    for rr = 1 : Nr
        SincKel = sinc(F_RCM(rr,aa) - NKel/2 : 1 : F_RCM(rr,aa) + NKel/2 - 1);
        SincKel = SincKel / sum(SincKel);
        kk = rr + N_RCM(rr, aa);
        kk = rem(kk, Nr);
        if (kk - NKel/2) > Nr
            nn = (kk - NKel/2 - Nr) : 1 : (kk + NKel/2 - 1 - Nr);
        elseif (kk + NKel/2 - 1) > Nr
            nn1 = (kk - NKel/2) : 1 : Nr;
            nn2 = 1 : 1 : (kk + NKel/2 - 1 - Nr);
            nn  = [nn1, nn2];
        elseif (kk + NKel/2 - 1) < 1
            nn = (kk - NKel/2 + Nr) : 1 : (kk + NKel/2 - 1 + Nr);
        elseif (kk - NKel/2) < 1
            nn1 = (kk - NKel/2 + Nr) : 1 : Nr;
            nn2 = 1 : 1 : (kk + NKel/2 - 1);
            nn  = [nn1, nn2];
        else
            nn = (kk - NKel/2) : 1 : (kk + NKel/2 - 1);
        end
        DATA_RCMC(rr, aa) = SincKel * DATA(nn, aa);
    end
end
DATA = DATA_RCMC; clear DATA_RCMC N_RCM F_RCM RCM;

% --- 3.4 方位压缩 ---
fprintf('方位压缩...\n');
NfftA   = 2^nextpow2(2 * Na);        % 方位 FFT 点数 (≥ 合成孔径长度)
Fnan    = Fdc + (-NfftA/2 : NfftA/2-1) / NfftA * PRF;      % 1xNfftA 行向量
fbase   = (-NfftA/2 : NfftA/2-1) / NfftA * PRF;             % 1xNfftA 行向量
D       = sqrt(1 - (lambda * Fnan / 2 / Vr).^2);            % 1xNfftA 徙动因子
Dc      = sqrt(1 - (lambda * Fdc / (2 * Vr))^2);
WinA    = ones(NfftA, 1);

Ha      = exp(1j * 4 * pi * Rrefs.' * D * fc / C) .* WinA.';  % (Nrx1)*(1xNfftA)=NrxNfftA
eta_c   = -lambda * Rrefs.' * Fdc / (2 * Vr^2 * Dc);           % Nrx1
Ha      = Ha .* exp(-1j * 2 * pi * eta_c * fbase);             % (NrxNfftA).*((Nrx1)*(1xNfftA))

temp    = fftshift(ifft(ifftshift(DATA, 2), [], 2), 2);
temp    = padarray(temp, [0 (NfftA - Na)/2], 0);
clear DATA;
for rr = 1 : Nr
    if mod(rr, 256) == 0
        fprintf('  AC: %d / %d\n', rr, Nr);
    end
    temp(rr, :) = fftshift(ifft(fft(ifftshift(temp(rr, :))) .* ifftshift(Ha(rr, :))));
end
DATA = temp(:, NfftA/2 - Na/2 + 1 : NfftA/2 + Na/2);
clear D Ha temp;

%% ===================== 4. 显示聚焦结果 =====================
img = abs(DATA);
imgNorm = img / max(img, [], 'all');

% 查找峰值位置
[~, idxR] = max(max(img, [], 2));
[~, idxC] = max(max(img, [], 1));
fprintf('\n峰值位置: 距离向第 %d 行, 方位向第 %d 列 (中心: %d, %d)\n', ...
    idxR, idxC, round(Nr/2), round(Na/2));

% 3 dB 分辨率
[pk, ~] = max(img(idxR, :));
halfA = find(img(idxR, :) >= pk / sqrt(2));
resA = (halfA(end) - halfA(1) + 1) * (Vr / PRF);
[pk, ~] = max(img(:, idxC));
halfR = find(img(:, idxC) >= pk / sqrt(2));
resR = (halfR(end) - halfR(1) + 1) * (C / 2 / Fr);
Ba_theory = 2 * Vr^2 * (Na/PRF) / (lambda * R0);   % 多普勒带宽理论值
fprintf('距离向 3dB 分辨率: %.2f m  (理论 %.2f m)\n', resR, C / 2 / Br);
fprintf('方位向 3dB 分辨率: %.2f m  (理论 %.2f m)\n', resA, Vr / Ba_theory);

% 图像显示
figure('Name','RD 聚焦结果 - 幅度','Color','w');
imagesc(imgNorm, [0, 0.05]); colormap(gray); axis image;
xlabel('方位向'); ylabel('距离向'); title('点目标 RD 成像结果 (线性)');
colorbar;

figure('Name','RD 聚焦结果 - dB','Color','w');
imagesc(db(imgNorm + 1e-10), [-40, 0]); colormap(gray); axis image;
xlabel('方位向'); ylabel('距离向'); title('点目标 RD 成像结果 (dB)');
colorbar;

% 中心区域放大
rHalf = 80; aHalf = 80;
rC = round(Nr/2); aC = round(Na/2);
figure('Name','中心点目标放大 (dB)','Color','w');
imagesc(db(imgNorm(rC-rHalf:rC+rHalf, aC-aHalf:aC+aHalf) + 1e-10), [-40, 0]);
colormap(gray); axis image;
xlabel('方位向'); ylabel('距离向'); title('中心区域放大 (验证聚焦)');
colorbar;

% 距离向 & 方位向切片
figure('Name','峰值切片','Color','w');
subplot(1,2,1);
plot(db(imgNorm(rC-rHalf:rC+rHalf, aC) + 1e-10), 'LineWidth', 1.5);
grid on; xlabel('距离单元'); ylabel('幅度 (dB)'); title('距离向切片');
subplot(1,2,2);
plot(db(imgNorm(rC, aC-aHalf:aC+aHalf) + 1e-10), 'LineWidth', 1.5);
grid on; xlabel('方位单元'); ylabel('幅度 (dB)'); title('方位向切片');

imwrite(uint8(255 * mat2gray(db(imgNorm + 1e-10), [-40, 0])), 'sim_point_RD.jpg');
fprintf('结果已保存: sim_point_RD.jpg\n');
