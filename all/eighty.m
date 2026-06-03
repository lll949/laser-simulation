
clear; clc; close all;

%% ==================== 物理常数和单位定义 ====================
nm = 1e-9; um = 1e-6; mm = 1e-3; cm = 1e-2; m = 1;
us = 1e-6; ms = 1e-3; s = 1;
c = 299792458; % 光速 [m/s]
h = 6.626e-34; % 普朗克常数 [J·s]

%% ==================== 模拟参数设置 ====================
lambda_pump = 808*nm;    % 泵浦波长
lambda_laser = 1064*nm;  % 激光波长
window = 4.2*mm;
N_grid = 512;
var = space('x', window, N_grid, 'y', window, N_grid);
dx = var.x_(2) - var.x_(1);
dy = var.y_(2) - var.y_(1);
X = var.x;
Y = var.y;
r = sqrt(X.^2 + Y.^2);
phi = atan2(Y, X);

% 增益介质参数
n_medium = 1.82;
l_gain = 10*mm;
L_cavity = 400*mm;
f_lens = 400*mm;
NA = 0.01;
% 物理参数
gain_medium.absorb_eff = 25;
sigma_em = 3e-16;      % 发射截面 [m^2]
tau = 230*us;            % 上能级寿命 [s]
beta_spon = 1e-6;        % 自发辐射耦合系数

%% ==================== 传播算子定义====================
k0_pump = 2*pi/lambda_pump;
k0_laser = 2*pi/lambda_laser;
k_medium_laser = n_medium * k0_laser;
k_medium_pump = n_medium * k0_pump;
max_k = 2*pi/lambda_laser * NA;
k_mask = (abs(2*pi*var.fx) < max_k) & (abs(2*pi*var.fy) < max_k);
k_mask = gpuArray(k_mask); 

% 真空传播
kz_laser_vac = sqrt(k0_laser^2 - (2*pi*var.fx).^2 - (2*pi*var.fy).^2);
kz_laser_vac = gpuArray(kz_laser_vac);
Propagation_vacuum = @(E, d) ifft2(fft2(E) .* exp(1i*kz_laser_vac*d) .* k_mask);

% 介质内传播（激光）
kz_laser_medium = sqrt(k_medium_laser^2 - (2*pi*var.fx).^2 - (2*pi*var.fy).^2);
kz_laser_medium = gpuArray(kz_laser_medium);
Propagation_in_gain = @(E, d) ifft2(fft2(E) .* exp(1i*kz_laser_medium*d) .* k_mask);

% 泵浦传播
kz_pump = sqrt(k0_pump^2 - (2*pi*var.fx).^2 - (2*pi*var.fy).^2);
kz_pump = gpuArray(kz_pump);
Propagation_pump = @(E, d) ifft2(fft2(E) .* exp(1i*kz_pump*d));
kz_pump_medium = sqrt(k_medium_pump^2 - (2*pi*var.fx).^2 - (2*pi*var.fy).^2);
kz_pump_medium = gpuArray(kz_pump_medium);
Propagation_pump_in_gain = @(E, d) ifft2(fft2(E) .* exp(1i*kz_pump_medium*d) .* k_mask);

% 透镜和光阑
Lens_phase = exp(-1i*k0_laser*(X.^2 + Y.^2)/(2*f_lens));
aperture_radius = 0.8*mm;
Aperture = (r < aperture_radius);
Lens_phase = gpuArray(Lens_phase);
Aperture = gpuArray(Aperture);

%% ==================== 激光模式====================
w0_lg = 361*um;
norm_factor_lg = sqrt(2/pi) * (1/w0_lg);
lg00 = sqrt(2/pi) * (1/w0_lg) * exp(-r.^2/w0_lg^2);
lg01_positive=LG(var,w0_lg,0,1,0,0);
lg01_negative=LG(var,w0_lg,0,-1,0,0);
lg00 = gpuArray(lg00);
lg01_positive = gpuArray(lg01_positive);
lg01_negative = gpuArray(lg01_negative);


%% ==================== 增益介质设置 ====================
gain_positions =40*mm;
gain_thickness =80*mm;
gain_segments_per_slice = 5;

total_gain_segments = length(gain_positions) * gain_segments_per_slice;

gain_media = struct('position', {}, 'z_start', {}, 'z_end', {}, ...
                    'dz', {}, 'segments', {}, 'pump_field', {});

for i = 1:length(gain_positions)
    gain_media(i).position = gain_positions(i);
    gain_media(i).z_start = gain_positions(i) - gain_thickness(i)/2;
    gain_media(i).z_end = gain_positions(i) + gain_thickness(i)/2;
    gain_media(i).dz = gain_thickness(i) / gain_segments_per_slice;
    gain_media(i).segments = gain_segments_per_slice;
    gain_media(i).dt = (gain_thickness(i) / gain_segments_per_slice) / (c / n_medium);
end
%% ==================== 定义四个不同手性的泵浦光（GPU）====================
gaussian = @(x, w) exp(-(x/w).^2);
ring_center = 323*um;
ring_width = 100*um;
l_global = -3;
phi_global = angle(gpuArray(X) + 1i*gpuArray(Y));
global_vortex_phase = exp(1i * l_global * phi_global);
pu =5*1e19;
r_ring = abs((gpuArray(X)) + 1i*(gpuArray(Y))) - ring_center;
ring_amplitude = gaussian(r_ring, ring_width);
pump_field =  sqrt(pu) .* ring_amplitude.*global_vortex_phase.* (cos(6*phi_global));
%% ==================== 主模拟循环：GPU优化版 ====================
N_iterations =300;
num_simulations = 1000;
mode_fractions_3d = zeros(num_simulations, N_iterations, 3);
mode_fraction_pos = zeros(1, N_iterations);
mode_fraction_neg = zeros(1, N_iterations);
E_saved = zeros(N_grid, N_grid, N_iterations);
E=zeros(N_grid, N_grid, N_iterations);
E_photon_pump = h * c / lambda_pump;
E_photon_laser = h * c / lambda_laser;

round_trip_time = 2 * L_cavity / c;
decay_factor = exp(-round_trip_time / tau);
I_sat = E_photon_laser / (sigma_em * tau);

% 预计算传播距离
sorted_pos = sort(gain_positions);
[~, sort_idx] = sort(gain_positions);
% 
for sim = 1:num_simulations   
    fprintf('运行第%d/%d轮仿真\n', sim, num_simulations);
% %    
    E_laser = gpuArray(zeros(N_grid, N_grid));
    N_inv_cumulative = gpuArray(zeros(N_grid, N_grid, length(gain_positions), gain_segments_per_slice));
    
    for iter = 1:N_iterations
        Pump_field_current = pump_field;
        current_z = 0;
        N_inv_new = gpuArray(zeros(N_grid, N_grid, length(gain_positions), gain_segments_per_slice));
%         fprintf('运行第%d/%d轮仿真\n', iter, N_iterations);
        % 泵浦吸收计算
        for g = 1:length(gain_positions)  
            gain = gain_media(g);
            if gain.z_start > current_z
               dist = gain.z_start - current_z;
               Pump_field_current = Propagation_pump(Pump_field_current, dist);
               current_z = gain.z_start;
            end
            for s = 1:gain.segments
                I1 = abs(Pump_field_current).^2;
                Pump_field_current = Pump_field_current .* exp(-0.5 * gain_medium.absorb_eff * gain.dz);
                I2 = abs(Pump_field_current).^2;
                delta_I = I1 - I2;
                delta_N = delta_I * dx * dy / E_photon_pump;
                N_inv_new(:, :, g, s) = delta_N * gain.dt;
                
                if s < gain.segments
                    Pump_field_current = Propagation_pump_in_gain(Pump_field_current, gain.dz);
                end
                current_z=current_z+ gain.dz;
            end
        end
        
        % 反转粒子数衰减与累积
        N_inv_cumulative = N_inv_cumulative * decay_factor + N_inv_new;
        
        % 激光放大过程
        for idx = 1:length(sort_idx)
            g = sort_idx(idx);
            gain = gain_media(g);
            
            if idx == 1
                prop_to_gain = gain.z_start;
            else
                prev_gain = gain_media(sort_idx(idx-1));
                prop_to_gain = gain.z_start - prev_gain.z_end;
            end
            
            if prop_to_gain > 0
                E_laser = Propagation_vacuum(E_laser, prop_to_gain);
            end
            
            for s = 1:gain.segments
                N_inv_current = N_inv_cumulative(:, :, g, s);
                gain_coeff = sigma_em * N_inv_current;
                amplitude_gain = exp(0.5 * gain_coeff * gain.dz);
                E_laser = E_laser .* amplitude_gain;
                spontaneous_power = beta_spon * (N_inv_current / tau) * E_photon_laser * gain.dt;
                E_spon = sqrt(spontaneous_power) .* exp(1i*2*pi*rand(size(E_laser)));
                E_laser = E_laser + E_spon;
                 laser_intensity = abs(E_laser).^2;
                % 反转粒子数消耗
                photons_consumed = laser_intensity / E_photon_laser * gain.dt * dx * dy;
                N_inv_cumulative(:, :, g, s) = max(0, N_inv_current - photons_consumed);
                
                if s < gain.segments
                    E_laser = Propagation_in_gain(E_laser, gain.dz);
                end
            end
        end
        
        % 腔镜和透镜
        last_gain = gain_media(sort_idx(end));
        prop_to_lens = L_cavity - last_gain.z_end;
        E_laser = Propagation_vacuum(E_laser, prop_to_lens);
        E_laser = E_laser .* Lens_phase;
        E_laser = Propagation_vacuum(E_laser, L_cavity/2);
         E_laser = E_laser .* Aperture;
         E_laser = Propagation_vacuum(E_laser, L_cavity/2);
        
        %% ========== 模式分解分析（在GPU上计算）==========
        E_saved(:, :, iter) = E_laser;
        Q = E_laser;
        QA00 = sum(Q .* conj(lg00), 'all');
        AA00 = sum(lg00 .* conj(lg00), 'all');
        QA = sum(Q .* conj(lg01_positive), 'all');
        QB = sum(Q .* conj(lg01_negative), 'all');
        AA = sum(lg01_positive .* conj(lg01_positive), 'all');
        BB = sum(lg01_negative .* conj(lg01_negative), 'all');
        QQ = sum(Q .* conj(Q), 'all');
        
        if QQ > 0
            frac_00 = gather(abs(QA00)^2 / (QQ * AA00));
            frac_pos = gather(abs(QA)^2 / (QQ * AA));
            frac_neg = gather(abs(QB)^2 / (QQ * BB));
        else
            frac_00 = 0; frac_pos = 0; frac_neg = 0;
        end
%         mode_fraction_pos(iter) = frac_pos;
%         mode_fraction_neg(iter) = frac_neg;
        mode_fractions_3d(sim, iter, 1) = frac_00;
        mode_fractions_3d(sim, iter, 2) = frac_pos;
        mode_fractions_3d(sim, iter, 3) = frac_neg;

    end
end

figure
filename = '2.gif';
for i = 1:N_iterations
    clf;
    current_00 = squeeze(mode_fractions_3d(:, i, 1));
    current_pos = squeeze(mode_fractions_3d(:, i, 2));
    current_neg = squeeze(mode_fractions_3d(:, i, 3));
    scatter3(current_pos, current_neg, current_00, ...
             30, 'r', 'filled', ...
             'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    xlabel('LG+1模式占比');
    ylabel('LG-1模式占比');
    zlabel('LG00模式占比');
    xlim([0, 1]);
    ylim([0, 1]);
    zlim([0, 1]);
    view(45,30);
    drawnow;
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % 写入GIF
    if i == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
%     pause(0.1);
end


% filename_phase = '相位演化LG1.gif';
% filename_intensity = '光场强度演化LG1.gif';
% for frame = 1:N_iterations
%     clf;
%     
%     % 子图1：模式占比散点图
%     subplot(2,2,1);
%     hold on;
%     if frame > 1
%         scatter(mode_fraction_pos(1:frame-1), mode_fraction_neg(1:frame-1), ...
%                 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
%     end
%     scatter(mode_fraction_pos(frame), mode_fraction_neg(frame), ...
%             100, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
%     
%     xlabel('LG+1模式占比');
%     ylabel('LG-1模式占比');
%     axis equal;
%     grid on;
%     xlim([0, max(1, max(mode_fraction_pos))]);
%     ylim([0, max(1, max(mode_fraction_neg))]);
%     hold off;
%     
%     % 子图2：光场强度分布
%     subplot(2,2,2);
%     current_intensity = abs(E_saved(:,:,frame)).^2;
%     imagesc(fftshift(var.x_/mm), fftshift(var.y_/mm), fftshift(current_intensity));
% %     axis equal tight;
%     title(sprintf('强度分布 (迭代%d)', frame));
%     xlabel('x (mm)'); ylabel('y (mm)');
%     colorbar;
%     colormap jet;
%     
%     % 单独保存子图2为GIF
% %     subplot(2,2,2);
% %     frame_intensity = getframe(gca);
% %     im_intensity = frame2im(frame_intensity);
% %     [imind_intensity, cm_intensity] = rgb2ind(im_intensity, 256);
% %     
% %     if frame == 1
% %         imwrite(imind_intensity, cm_intensity, filename_intensity, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
% %     else
% %         imwrite(imind_intensity, cm_intensity, filename_intensity, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
% %     end
% %     
%     % 子图3：相位分布
%     subplot(2,2,3);
%     plot_complex_field(E_saved(:,:,frame), var.x_/mm, var.y_/mm);
%     title(sprintf('相位分布 (迭代%d)', frame));
%     
%     % 单独保存子图3为GIF
% %     subplot(2,2,3);
% %     frame_phase = getframe(gca);
% %     im_phase = frame2im(frame_phase);
% %     [imind_phase, cm_phase] = rgb2ind(im_phase, 256);
% %     
% %     if frame == 1
% %         imwrite(imind_phase, cm_phase, filename_phase, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
% %     else
% %         imwrite(imind_phase, cm_phase, filename_phase, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
% %     end
%     
%     % 子图4：反转粒子数分布（第一个截面）
%     subplot(2,2,4);
%     plot(1:frame, mode_fraction_pos(1:frame), 'r-', 'LineWidth', 2, 'DisplayName', 'LG01');
%     hold on;
%     plot(1:frame, mode_fraction_neg(1:frame), 'b-', 'LineWidth', 2, 'DisplayName', 'LG0-1');
%     hold off;
%     xlabel('迭代次数');
%     ylabel('模式占比');
%     title('模式占比演化');
%     legend('Location', 'best');
%     grid on;
%     xlim([1, N_iterations]);
%     ylim([0, max(1, max([mode_fraction_pos, mode_fraction_neg]))]);
%     
%     drawnow;
%     pause(0.1);
%  end

%  frames_to_save = [3, 12, 40];   % 需要保存的帧号
% 
% fig = figure('Position', [100 100 1200 900]);   % 放大窗口
% 
% 
% for i = 1:length(frames_to_save)
%     frame = frames_to_save(i);
%     plot_complex_field(E_saved(:,:,frame), var.x_/mm, var.y_/mm);
%     axis equal tight;
%     axis off;
%     exportgraphics(fig, sprintf('phase_frame_%d.png', frame), 'Resolution', 600);
% end
% 
% fig = figure('Position', [100 100 1200 900]);   % 放大窗口
% 
% 
% for frame = frames_to_save
%     clf(fig);   % 清空图形窗口
%     imagesc(fftshift(var.x_)/mm, fftshift(var.y_)/mm, fftshift(abs(E_saved(:,:,frame)).^2));
%     axis equal tight;
%     axis off;
%     colormap jet;
%     exportgraphics(fig, sprintf('phase_frame_1%d.png', frame), 'Resolution', 600);
%    fprintf('LG01: %f, LG0-1: %f\n', mode_fraction_pos(frame), mode_fraction_neg(frame));
% end


iterations_to_show = [3, 9, 300];
colors = {'r', 'g', 'b'};  % 红、绿、蓝分别对应5、10、25次迭代
markers = {'o', 's', '^'};  % 不同形状：圆形、方形、三角形
labels = {'第5次迭代', '第10次迭代', '第25次迭代'};

% 创建图形
figure;

% 存储所有数据用于统一坐标轴范围
all_frac_00 = [];
all_frac_pos = [];
all_frac_neg = [];
all_data = table();
% % 提取并存储每次迭代的数据
for idx = 1:length(iterations_to_show)
    iter = iterations_to_show(idx);
    
    % 提取该次迭代所有仿真轮次的数据
    frac_00 = squeeze(mode_fractions_3d(:, iter, 1));
    frac_pos = squeeze(mode_fractions_3d(:, iter, 2));
    frac_neg = squeeze(mode_fractions_3d(:, iter, 3));
    T = table(repmat(iter, length(frac_00), 1), ...  % 迭代次数列
              frac_00, frac_pos, frac_neg, ...
              'VariableNames', {'Iteration', 'LG00_Fraction', 'LGpos_Fraction', 'LGneg_Fraction'});
    all_data = [all_data; T];
    % 存储到总数据中
%     all_frac_00 = [all_frac_00; frac_00];
%     all_frac_pos = [all_frac_pos; frac_pos];
%     all_frac_neg = [all_frac_neg; frac_neg];
    
    % 绘制散点图
    scatter3(frac_pos, frac_neg, frac_00, ...
             15, colors{idx}, markers{1}, ...
             'filled', ...
             'MarkerEdgeColor', 'k', ...
             'LineWidth', 0.2, ...
             'DisplayName', labels{idx});
    hold on;
    xlabel('LG+1 模式占比', 'FontSize', 12, 'FontWeight', 'bold');
    
    ylabel('LG-1 模式占比', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('LG00 模式占比', 'FontSize', 12, 'FontWeight', 'bold');
title('不同迭代次数的模式占比三维分布对比', 'FontSize', 14, 'FontWeight', 'bold');

% 设置坐标轴范围
xlim([0, 1]);
ylim([0, 1]);
zlim([0, 1]);

% 添加网格和视角
grid on;
box on;
view(45, 30);  % 设置视角
end
saveas(gcf, '模式占比三维对比图2.png');

writetable(all_data, '模式占比数据100528.xlsx');

