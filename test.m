function [] = test(start,stop)
[y,Fs] = audioread('jee.mp3');
player = audioplayer(y,Fs);
start = player.SampleRate*start;
stop = player.SampleRate*stop;

playblocking(player,[start,stop]);

% y = y(:,1);
% dt = 1/Fs;
% t = 0:dt:(length(y)*dt)-dt;
% plot(t,y); xlabel('Seconds'); ylabel('Amplitude');