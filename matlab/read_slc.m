% ========= 读取 S1A SLC 数据并显示幅度图 =========
% Sentinel-1 SLC 为 complex int16 TIFF (每像素 4 字节 = real_int16 + imag_int16),
% MATLAB 内置 Tiff.read 不支持该复数格式,
% 因此通过 Tiff 对象获取 strip 偏移量, 再用 fread 读原始字节并 typecast 为 int16。

slc_dir = 'C:\Users\Miss_\Desktop\Sentinel-L0条带\S1A_S2_SLC__1SDV_20260621T014858_20260621T014917_065061_083350_FCA0.SAFE';

%% 1. 定位 TIFF 文件和 XML 文件
d = dir(fullfile(slc_dir, 'measurement', '*.tiff'));
if isempty(d)
    d = dir(fullfile(slc_dir, '**', '*.tiff'));
end
if isempty(d)
    error('未找到 .tiff 文件，请检查 slc_dir 路径是否正确！');
end
slc_file = fullfile(d(1).folder, d(1).name);
[~, base] = fileparts(slc_file);
xml_file = fullfile(slc_dir, 'annotation', [base '.xml']);
fprintf('正在处理文件: %s\n', base);

%% 2. 从 XML 获取图像尺寸
try
    xml_doc = xmlread(xml_file);
    nRows = str2double(xml_doc.getElementsByTagName('numberOfLines').item(0).getTextContent());
    nCols = str2double(xml_doc.getElementsByTagName('numberOfSamples').item(0).getTextContent());
catch
    nRows = 30749;
    nCols = 20099;
    warning('从 XML 读取尺寸失败，已使用默认尺寸兜底。');
end
fprintf('图像真实尺寸: %d 行(脉冲数) x %d 列(采样数)\n', nRows, nCols);

%% 3. 通过 Tiff 对象获取 strip 偏移量与字节数
fprintf('正在解析 TIFF strip 信息...\n');
t = Tiff(slc_file, 'r');
imgW   = getTag(t, 'ImageWidth');
imgH   = getTag(t, 'ImageLength');
stripOffsets    = getTag(t, 'StripOffsets');
stripByteCounts = getTag(t, 'StripByteCounts');
close(t);
fprintf('TIFF: %d x %d, strip 数: %d, 每 strip %d 字节\n', ...
    imgW, imgH, numel(stripOffsets), double(stripByteCounts(1)));

%% 4. 抽样读取并转换 (防止内存不足)
% SLC 数据按 complex int16 存储: 每像素 4 字节 = [real_int16, imag_int16] (小端)
step = 4;                              % 每隔 4 行/列取 1 点, 数据量缩小 16 倍
read_rows = 1:step:nRows;
nR = numel(read_rows);
nC = numel(1:step:nCols);
amp = zeros(nR, nC, 'single');
fprintf('正在抽样读取 SLC 数据 (%d 行 x %d 列)...\n', nR, nC);

fid = fopen(slc_file, 'r');
bytesPerRow = double(stripByteCounts(1));    % 每行字节数
colIdx = 1:step:nCols;                        % 抽样列
for i = 1:nR
    r = read_rows(i);
    fseek(fid, double(stripOffsets(r)), 'bof');
    row_bytes = fread(fid, bytesPerRow, 'uint8=>uint8');
    row_i16 = typecast(row_bytes, 'int16');   % 交错: real, imag, real, imag, ...
    real_idx = colIdx * 2 - 1;                % 实部在奇数位
    imag_idx = colIdx * 2;                    % 虚部在偶数位
    real_s = single(row_i16(real_idx));
    imag_s = single(row_i16(imag_idx));
    amp(i, :) = sqrt(real_s.^2 + imag_s.^2);
end
fclose(fid);
fprintf('数据读取完成！首像素幅度: %.4f (期望 ~1.0)\n', amp(1,1));

%% 6. 计算幅度(dB)并显示
amp_dB = 20 * log10(amp + 1e-6);
clear amp;

figure('Name', 'SLC 幅度图', 'Color', 'w');
vmin = prctile(amp_dB(:), 2);
vmax = prctile(amp_dB(:), 98);
imagesc(amp_dB, [vmin, vmax]);
colormap gray;
c = colorbar;
c.Label.String = '幅度 (dB)';
title(sprintf('SLC 幅度图 (抽样率 1/%d) - %s', step, base), 'Interpreter', 'none');
xlabel(sprintf('距离向 (Range) - 抽样后点数: %d', size(amp_dB,2)));
ylabel(sprintf('方位向 (Azimuth) - 抽样后点数: %d', size(amp_dB,1)));
%axis image;
axis xy;   % 让 Y 轴坐标从小到大（原点在左下角），符合常规阅读习惯
axis on;   % 强制打开坐标轴刻度和边框
saveas(gcf, 'SLC_amplitude.jpg');
fprintf('幅度图已保存: SLC_amplitude.jpg\n');

%% 7. 提取并保存参数
params = struct();
params.file            = base;
params.numberOfLines   = nRows;        % 方位向脉冲数
params.numberOfSamples = nCols;        % 距离向采样点数
params.step            = step;
params.meanAmp_dB      = mean(amp_dB(:), 'omitnan');
params.maxAmp_dB       = max(amp_dB(:));

if isfile(xml_file)
    tags = {'rangePixelSpacing', 'azimuthPixelSpacing', ...
            'azimuthTimeInterval', 'rangeSamplingRate', ...
            'radarFrequency', 'prf', 'slantRangeTime', 'pass'};
    for k = 1:numel(tags)
        n = xml_doc.getElementsByTagName(tags{k}).item(0);
        if ~isempty(n)
            val = char(n.getTextContent());
            num_val = str2double(val);
            if isnan(num_val)
                params.(tags{k}) = val;
            else
                params.(tags{k}) = num_val;
            end
        end
    end
    params.wavelength = 299792458 / params.radarFrequency;
end

disp('==== 提取并保存的核心参数 ====');
disp(params);
save('slc_params.mat', 'params');
fprintf('参数已成功保存至 slc_params.mat\n');
%% ===== 交互式切图（让鼠标帮你找最亮的点） =====
figure('Name', '请在大图上点击 3 个亮目标');
% 显示抽样后的大图
imagesc(amp_dB, [prctile(amp_dB(:), 2), prctile(amp_dB(:), 98)]); 
colormap gray; axis xy; axis on;
title('请在图中点击 3 个不同的【明亮区域】或【边缘交界处】，按回车结束');

disp('等待您在图上点击 3 次...');
[cols, rows] = ginput(3); % 鼠标在图上点3下

roi_size = 512;
% 注意：这里直接把鼠标点击的坐标换算到全分辨率
centers_full = round([rows, cols] * step); 
% ===== 重新打开文件，获取新的有效 fid =====
fid = fopen(slc_file, 'r');
if fid == -1, error('重新打开 SLC 文件失败！'); end

for i = 1:3
    start_r = round(centers_full(i, 1) - roi_size/2);
    start_c = round(centers_full(i, 2) - roi_size/2);
    
    start_r = max(1, min(start_r, nRows - roi_size));
    start_c = max(1, min(start_c, nCols - roi_size));
    
    roi_complex = zeros(roi_size, roi_size, 'single');
    
    for r_idx = 1:roi_size
        current_row = start_r + r_idx - 1;
        fseek(fid, double(stripOffsets(current_row)), 'bof');
        row_bytes = fread(fid, bytesPerRow, 'uint8=>uint8');
        row_i16 = typecast(row_bytes, 'int16');
        row_complex_full = complex(row_i16(1:2:end), row_i16(2:2:end));
        roi_complex(r_idx, :) = row_complex_full(start_c : start_c + roi_size - 1);
    end
    
    save_name = sprintf('ROI_%d_complex.mat', i);
    save(save_name, 'roi_complex');
    
    figure; 
    imagesc(20*log10(abs(roi_complex))); 
    colormap gray; axis xy; axis on; colorbar;
    title(sprintf('ROI %d (起点: %d, %d)', i, start_r, start_c));
    saveas(gcf, sprintf('ROI_%d_preview.png', i));
end
fclose(fid);
disp('🎉 交互式切图完成！');