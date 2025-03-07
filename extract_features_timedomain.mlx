function features = extract_features(emg_segment)
    mav = mean(abs(emg_segment)); % Mean Absolute Value
    zc = sum(diff(sign(emg_segment)) ~= 0); % Zero Crossings
    ssc = sum(diff(sign(diff(emg_segment))) ~= 0); % Slope Sign Changes
    wl = sum(abs(diff(emg_segment))); % Waveform Length
    rms = sqrt(mean(emg_segment.^2)); % Root Mean Square
    var_emg = var(emg_segment); % Variance of EMG
    features = [mav, zc, ssc, wl, rms, var_emg];
end

