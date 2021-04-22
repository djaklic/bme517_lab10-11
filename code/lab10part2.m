clc
clear
patientA = load('Patient_A.mat').data;
patientB = load('Patient_B.mat').data;

% Berger bands      1-70	
% Fast gamma        70-150	
% Ripples           150-250	
% Fast Ripples      250-600	
% Units             1000-2000	

% HFO: 100-500 Hz
% 20-150 ms stands out

% Patient A, seizure can be seen on channel 3, sample 2480000
% Patient B, seizure can be seen on channel 2, sample 2800000

[patientA_detect, patientA_filtered] = mySeizureDetectorHFO(patientA, 3);
[patientB_detect, patientB_filtered] = mySeizureDetectorHFO(patientB, 2);

plotStuffHFO(patientA, patientA_filtered, 3, [2250000 3000000]);
%plotStuffHFO(patientB, patientB_filtered, 2, [2500000 3500000]);

function [detected, signal_filtered] = mySeizureDetectorHFO(signal, channel)
    detected = false;
    [signal_filtered, filter] = bandpass(signal(channel,:), [100 500], 3000);
    signal_filtered = filtfilt(filter, signal_filtered);
end

function [] = plotStuffHFO(signal_raw, signal_filtered, channel, range)
    f1 = figure;
    subplot(2,1,1)
    hold on
    plot(signal_raw(channel,:));
    plot(signal_filtered);
    legend('raw signal', 'HFO')
    xlim([0 5400000]);
    hold off
    subplot(2,1,2)
    hold on
    %plot(signal_raw(channel,range));
    %plot(signal_filtered(:,range));
    plot(signal_raw(channel,:));
    plot(signal_filtered);
    xlim(range);
    ylim([-500 500]);
    hold off
    
    ax=axes(f1,'visible','off');
    ax.XLabel.Visible='on';
    ax.YLabel.Visible='on';
    xlabel(ax, 'Time bin (3 kHz)');
    ylabel(ax, 'Voltage (uV)');
    %legend(ax, 'signal', 'detection');
    saveas_ = '../figures/lab10_ExOfHFO';
    savefig(append(saveas_, '.fig'))
    saveas(f1, append(saveas_, '.jpg'))
end