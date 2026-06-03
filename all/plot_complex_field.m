function plot_complex_field(E, x, y)
    amplitude = abs(E);
    amplitude_normalized = amplitude / max(amplitude(:));
    phase = mod(angle(E), 2*pi);
    H = phase / (2*pi);
    S = ones(size(E));
    V = amplitude_normalized;
    HSV = cat(3, H, S, V);
    RGB = hsv2rgb(HSV);
    imagesc(fftshift(x), fftshift(y), fftshift(RGB));
    axis equal tight;
    xlabel('x (mm)'); ylabel('y (mm)');
end
