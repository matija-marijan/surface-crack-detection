clear
close all
clc

%% Ucitavanje prethodnih promenljivih

% save uporedna_analiza.mat
% load uporedna_analiza.mat

%% Ucitavanje slika

neg_imds = imageDatastore('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Dataset\Gray\Negative\');
pos_imds = imageDatastore('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Dataset\Gray\Positive\');

% Broj slika po klasi - Klase su balansirane
N = numel(neg_imds.Files);

% Velicina slike
M = length(readimage(neg_imds, 1));

%% Izdvajanje obelezja

num_ob = 16;
ob_neg = zeros(N, num_ob);
ob_pos = zeros(N, num_ob);
 
% Negative obelezja
for i = 1 : N
    im = readimage(neg_imds, i);
    
    % Statisticka obelezja
    ob_neg(i, 1) = mean2(im);
    ob_neg(i, 2) = std2(im);
    ob_neg(i, 3) = median(im, 'all');
    ob_neg(i, 4) = entropy(im);
    ob_neg(i, 5) = rms(im, 'all');

    % Morfoloska obelezja
    thresh = 125;
    im_T = im > thresh;
    im_T = 1 - im_T;
    se = ones(3);
    im_open = imopen(im_T, se);
    se = ones(5);
    im_close = imclose(im_open, se);
    log_im_close = logical(im_close);
    [objects, num_objects] = bwlabel(log_im_close);
    object_area = bwarea(log_im_close);

    ob_neg(i, 6) = num_objects;
    ob_neg(i, 7) = object_area;
%     ob_neg(i, 8) = mean2(255 * (1 - im_T)) / mean2(im);
    ob_neg(i, 8) = mean2(255 * (1 - log_im_close)) / mean2(im);
    ob_neg(i, 9) = length(nonzeros(im_T));

    % Frekvencijska obelezja
    im_shift_fft = abs(fftshift(fft2(im)));
    fc = 0.5;
    h = (M - 1) / 2;
    [x, y] = meshgrid(-h : h, -h : h);
    mg = sqrt((x / h) .^ 2 + (y / h) .^ 2);
    lp = double(mg <= fc);
    LP = im_shift_fft .* lp;
    HP = im_shift_fft - LP;
    LPno = nonzeros(LP);
    HPno = nonzeros(HP);
    LF = mean(LPno);
    HF = mean(HPno);
    odnos = LF / HF;        % odnos = sum(sum(HP)) / sum(sum(LP));

    ob_neg(i, 10) = odnos;
    ob_neg(i, 11) = LF;
    ob_neg(i, 12) = HF;
    ob_neg(i, 13) = mean2(im_shift_fft);
    ob_neg(i, 14) = median(im_shift_fft, 'all');

    % Hough obelezja
    im_edge = edge(im_close, 'canny');
    [H, T, R] = hough(im_edge);
    P = houghpeaks(H, 100, 'threshold', ceil(0.85 * max(H(:))));
    if any(any(H))
        numpeaks = length(P);
    else
        numpeaks = 0;
    end

    P = houghpeaks(H, 5, 'threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(im_edge, T, R, P, 'FillGap', 10, 'MinLength', 15);
    max_len = 0;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        len = norm(lines(k).point1 - lines(k).point2);
        if (len > max_len)
            max_len = len;
        end
    end

    ob_neg(i, 15) = numpeaks;
    ob_neg(i, 16) = max_len;

end

% Positive obelezja
for i = 1 : N
    im = readimage(pos_imds, i);
    
    % Statisticka obelezja
    ob_pos(i, 1) = mean2(im);
    ob_pos(i, 2) = std2(im);
    ob_pos(i, 3) = median(im, 'all');
    ob_pos(i, 4) = entropy(im);
    ob_pos(i, 5) = rms(im, 'all');

    % Morfoloska obelezja
    thresh = 125;
    im_T = im > thresh;
    im_T = 1 - im_T;
    se = ones(3);
    im_open = imopen(im_T, se);
    se = ones(5);
    im_close = imclose(im_open, se);
    log_im_close = logical(im_close);
    [objects, num_objects] = bwlabel(log_im_close);
    object_area = bwarea(log_im_close);

    ob_pos(i, 6) = num_objects;
    ob_pos(i, 7) = object_area;
%     ob_pos(i, 8) = mean2(255 * (1 - im_T)) / mean2(im);
    ob_pos(i, 8) = mean2(255 * (1 - log_im_close)) / mean2(im);
    ob_pos(i, 9) = length(nonzeros(im_T));

    % Frekvencijska obelezja
    im_shift_fft = abs(fftshift(fft2(im)));
    fc = 0.5;
    h = (M - 1) / 2;
    [x, y] = meshgrid(-h : h, -h : h);
    mg = sqrt((x / h) .^ 2 + (y / h) .^ 2);
    lp = double(mg <= fc);
    LP = im_shift_fft .* lp;
    HP = im_shift_fft - LP;
    LPno = nonzeros(LP);
    HPno = nonzeros(HP);
    LF = mean(LPno);
    HF = mean(HPno);
    odnos = LF / HF;        % odnos = sum(sum(HP)) / sum(sum(LP));

    ob_pos(i, 10) = odnos;
    ob_pos(i, 11) = LF;
    ob_pos(i, 12) = HF;
    ob_pos(i, 13) = mean2(im_shift_fft);
    ob_pos(i, 14) = median(im_shift_fft, 'all');

    % Hough obelezja
    im_edge = edge(im_close, 'canny');
    [H, T, R] = hough(im_edge);
    P = houghpeaks(H, 100, 'threshold', ceil(0.85 * max(H(:))));
    if any(any(H))
        numpeaks = length(P);
    else
        numpeaks = 0;
    end

    P = houghpeaks(H, 5, 'threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(im_edge, T, R, P, 'FillGap', 10, 'MinLength', 15);
    max_len = 0;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        len = norm(lines(k).point1 - lines(k).point2);
        if (len > max_len)
            max_len = len;
        end
    end

    ob_pos(i, 15) = numpeaks;
    ob_pos(i, 16) = max_len;

end

ob = [ob_neg; ob_pos];

%% Normalizacija obelezja

ob_norm = zeros(size(ob));

ob_mean = zeros(num_ob, 1);
ob_std = zeros(num_ob, 1);

for i = 1 : num_ob
    ob_mean(i) = mean(ob(:, i));
    ob_std(i) = std(ob(:, i));
end

for i = 1 : num_ob
    ob_norm(:, i) = (ob(:, i) - ob_mean(i)) / ob_std(i);
end

%% Cuvanje svih obelezja

neg_norm = ob_norm(1 : N, :)';
pos_norm = ob_norm(N + 1 : end, :)';
% writematrix(neg_norm, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_norm_all.csv');
% writematrix(pos_norm, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_norm_all.csv');

%% Box plot analiza
close all

f = figure;
subplot(211)
boxplot(ob_norm(1 : N, :))
title('Bez pukotine')
grid on
% xlabel('Redni broj obelezja')
xlabel(' ')
ylabel('Normirana vrednost [~]')
% ylim([-7 19])
% axis tight
subplot(212)
boxplot(ob_norm(N + 1 : end, :))
title('Sa pukotinom')
grid on
xlabel('Redni broj obelezja')
ylabel('Normirana vrednost [~]')
% ylim([-7 19])
% axis tight
% axis fit
set(gcf, 'Color', 'w')
f.Position(3 : 4) = [800, 500];
% sgtitle('Box plot analiza')
% export_fig(1, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\boxplot.png', '-painters', '-r300');

%% Matrica kros-korelacije i raspodele
close all

varNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", ...
    "10", "11", "12", "13", "14", "15", "16"];

figure
corrplot(ob_norm, 'VarNames', varNames)
set(findall(gcf, 'type', 'axes'), 'XTickLabel', [])
set(findall(gcf, 'type', 'axes'), 'YTickLabel', [])
set(gcf, 'Color', 'w')
% title('Matrica kros-korelacije')
title('')
% ylabel('Redni broj obelezja')
% xlabel('Redni broj obelezja')
% axis tight
% export_fig(1, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\corrplot.png', '-painters', '-r300');

%% Matrica kros-korelacije
close all

rho = corrcoef(ob_norm);

figure
h = heatmap(rho);
h.XDisplayLabels = varNames;
h.YDisplayLabels = varNames;
h.CellLabelFormat = '%.3f';
ylabel('Redni broj obelezja')
xlabel('Redni broj obelezja')
set(gcf, 'Color', 'w')
% colormap gray
% export_fig(1, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\corrmatrix.png', '-painters', '-r300');

% figure
% imshow(rho, [])
% figure
% plotmatrix(rho, 'b.')

%% Matrica raspodela i histograma
close all

f = figure;
[~, ax] = plotmatrix(ob_norm(1 : 25 : end, :));
axis square
set(findall(gcf, 'type', 'axes'), 'XTickLabel', [])
set(findall(gcf, 'type', 'axes'), 'YTickLabel', [])
set(gcf, 'Color', 'w')
for i = 1 : length(varNames)
    ax(i, 1).YLabel.String = varNames(i);
    ax(length(varNames), i).XLabel.String = varNames(i);
end
f.Position(3 : 4) = [1000, 1000];
% xlabel('Redni broj obelezja')
% ylabel('Redni broj obelezja')
% export_fig(1, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\plotmatrix.png', '-painters', '-r300');

%% Histogrami obelezja
close all

for i = 1 : num_ob
    figure
    histogram(ob_norm(1 : N, i), 50, 'Normalization', 'pdf', 'FaceColor', 'b')
    hold all
    histogram(ob_norm(N + 1 : end, i), 50, 'Normalization', 'pdf', 'FaceColor', 'r')
    hold off
    grid on
    ylabel('Funkcija gustine verovatnoce')
    xlabel('Normirana vrednost obelezja')
%     title(['Obelezje ' num2str(i)])
    legend('Bez pukotine', 'Sa pukotinom')
    set(gcf, 'Color', 'w')
%     export_fig(i, ['C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\hist', num2str(i), '.png'], '-painters', '-r300')
end

% 8 bolje od 9, ima smisla jer i na positive slikama ima samo nekoliko
% linija koje doprinose HF snazi, ali LF se razlikuje zbog crnih procepa

%% Prikaz izabranih obelezja
close all

% ind_ob = [3, 8, 13];
ind_ob = [1, 4, 5];
% ind_ob = [2, 7, 16]; % final (?)
pos_prikaz = pos_norm(ind_ob, :);
neg_prikaz = neg_norm(ind_ob, :);

ob1_min = min(min(neg_prikaz(1, :)), min(pos_prikaz(1, :)));
ob2_min = min(min(neg_prikaz(2, :)), min(pos_prikaz(2, :)));
ob3_min = min(min(neg_prikaz(3, :)), min(pos_prikaz(3, :)));

ob1_max = max(max(neg_prikaz(1, :)), max(pos_prikaz(1, :)));
ob2_max = max(max(neg_prikaz(2, :)), max(pos_prikaz(2, :)));
ob3_max = max(max(neg_prikaz(3, :)), max(pos_prikaz(3, :)));

% xlim([ob1_min - 0.5, ob1_max + 0.5])
% ylim([ob2_min - 0.5, ob2_max + 0.5])
% zlim([ob3_min - 0.5, ob3_max + 0.5])

figure
plot3(neg_prikaz(1, :), neg_prikaz(2, :), neg_prikaz(3, :), 'bo', ...
    pos_prikaz(1, :), pos_prikaz(2, :), pos_prikaz(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
axis square; grid on; box on; axis tight
set(gcf, 'Color', 'w')
% view([0 -90 0])
% title('Prikaz obelezja u prostoru')
% export_fig(1, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\2_11.png', '-painters', '-r300')

figure
plot3(ob_neg(:, ind_ob(1)), ob_neg(:, ind_ob(2)), ob_neg(:, ind_ob(3)), 'bo', ...
    ob_pos(:, ind_ob(1)), ob_pos(:, ind_ob(2)), ob_pos(:, ind_ob(3)), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
axis square; grid on; box on; axis tight
set(gcf, 'Color', 'w')

figure
plot(neg_prikaz(1, :), neg_prikaz(2, :), 'bo', ...
    pos_prikaz(1, :), pos_prikaz(2, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
axis square; grid on; box on
set(gcf, 'Color', 'w')
xlim([ob1_min - 0.5, ob1_max + 0.5])
ylim([ob2_min - 0.5, ob2_max + 0.5])

figure
plot(neg_prikaz(1, :), neg_prikaz(3, :), 'bo', ...
    pos_prikaz(1, :), pos_prikaz(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(3))])
axis square; grid on; box on
set(gcf, 'Color', 'w')
xlim([ob1_min - 0.5, ob1_max + 0.5])
ylim([ob3_min - 0.5, ob3_max + 0.5])

figure
plot(neg_prikaz(2, :), neg_prikaz(3, :), 'bo', ...
    pos_prikaz(2, :), pos_prikaz(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel(['Obelezje ', num2str(ind_ob(2))])
ylabel(['Obelezje ', num2str(ind_ob(3))])
axis square; grid on; box on
set(gcf, 'Color', 'w')
xlim([ob2_min - 0.5, ob2_max + 0.5])
ylim([ob3_min - 0.5, ob3_max + 0.5])

%% PCA analiza
close all

[U, S, V] = svd(ob_norm, 'econ');
L = diag(S) .^ 2;

figure
subplot(211)
stem(L, 'x')
ylim([0 325000])
grid on
% title('PCA')
xlabel('k')
ylabel('\lambda_k')
subplot(212)
stem(cumsum(L) / sum(L), 'x')
ylim([0 1.1])
grid on
% title('PCA')
xlabel('k')
ylabel('Normirana kumulativna suma')
set(gcf, 'Color', 'w')
% export_fig(1, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\PCA.png', '-painters', '-r300')

% PCA redukcija dimenzija
red_PCA = 3;
ob_PCA = ob_norm * V(:, 1 : red_PCA);
neg_PCA = ob_PCA(1 : N, :)';
pos_PCA = ob_PCA(N + 1 : end, :)';

figure
plot3(neg_PCA(1, :), neg_PCA(2, :), neg_PCA(3, :), 'bo', ...
    pos_PCA(1, :), pos_PCA(2, :), pos_PCA(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('PCA obelezje 1')
ylabel('PCA obelezje 2')
zlabel('PCA obelezje 3')
% title('Prikaz obelezja u prostoru')
axis square; box on; grid on
set(gcf, 'Color', 'w')
% export_fig(2, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Izvestaj\Slike za izvestaj\PCA_obelezja.png', '-painters', '-r300')

figure
plot(neg_PCA(1, :), neg_PCA(2, :), 'bo', pos_PCA(1, :), pos_PCA(2, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('PCA obelezje 1')
ylabel('PCA obelezje 2')
axis square; box on; grid on
set(gcf, 'Color', 'w')

figure
plot(neg_PCA(1, :), neg_PCA(3, :), 'bo', pos_PCA(1, :), pos_PCA(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('PCA obelezje 1')
ylabel('PCA obelezje 3')
axis square; box on; grid on
set(gcf, 'Color', 'w')

figure
plot(neg_PCA(2, :), neg_PCA(3, :), 'bo', pos_PCA(2, :), pos_PCA(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('PCA obelezje 2')
ylabel('PCA obelezje 3')
axis square; box on; grid on
set(gcf, 'Color', 'w')

% writematrix(neg_PCA, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_PCA.csv');
% writematrix(pos_PCA, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_PCA.csv');

figure
heatmap(abs(V(:, 1:3)), 'ColorScaling', 'scaledcolumns');
colorbar off
xlabel('PCA obelezje')
ylabel('Izvorno obelezje')
set(gcf, 'Color', 'w')

%% LDA analiza
close all

m0 = mean(ob)';
m1 = mean(ob(1 : N, :))';
m2 = mean(ob(N + 1 : end, :))';

S1 = cov(ob(1 : N, :))';
S2 = cov(ob(N + 1 : end, :))';

Sw = S1 + S2;
Sb = (m1 - m0) * (m1 - m0)' + (m2 - m0) * (m2 - m0)';

delta = 0.01;
reg_Sw = (1 / (1 + delta)) * (Sw + delta * eye(length(Sw)));
[Fi, l] = eigs(reg_Sw ^ -1 * Sb, 16);
l = diag(real(l));

figure
subplot(211)
stem(l, 'x')
grid on
ylabel('\lambda_k')
xlabel('k')
subplot(212)
stem(cumsum(l) / sum(l), 'x')
grid on
% title('LDA: Normirana kumulativna suma')
ylabel('Normirana kumulativna suma')
xlabel('k')
set(gcf, 'Color', 'w')

% LDA redukcija dimenzija
red_LDA = 3;       
ob_LDA = ob * Fi(:, 1 : red_LDA);
neg_LDA = ob_LDA(1 : N, :)';
pos_LDA = ob_LDA(N + 1 : end, :)';

figure
plot3(neg_LDA(1, :), neg_LDA(2, :), neg_LDA(3, :), 'bo', ...
    pos_LDA(1, :), pos_LDA(2, :), pos_LDA(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('Obelezje 1')
ylabel('Obelezje 2')
zlabel('Obelezje 3')
% title('Prikaz obelezja u prostoru')
axis square; grid on, box on
set(gcf, 'Color', 'w')

figure
plot(neg_LDA(1, :), neg_LDA(2, :), 'bo', pos_LDA(1, :), pos_LDA(2, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('LDA obelezje 1')
ylabel('LDA obelezje 2')
axis square; box on; grid on
set(gcf, 'Color', 'w')

figure
plot(neg_LDA(1, :), neg_LDA(3, :), 'bo', pos_LDA(1, :), pos_LDA(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('LDA obelezje 1')
ylabel('LDA obelezje 3')
axis square; box on; grid on
set(gcf, 'Color', 'w')

figure
plot(neg_LDA(2, :), neg_LDA(3, :), 'bo', pos_LDA(2, :), pos_LDA(3, :), 'rx')
legend('Bez pukotine', 'Sa pukotinom')
xlabel('LDA obelezje 2')
ylabel('LDA obelezje 3')
axis square; box on; grid on
set(gcf, 'Color', 'w')

% writematrix(neg_LDA, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_LDA.csv');
% writematrix(pos_LDA, 'C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_LDA.csv');

figure
heatmap(abs(Fi(:, 1:3)), 'ColorScaling', 'scaledcolumns');
colorbar off
xlabel('LDA obelezje')
ylabel('Izvorno obelezje')
set(gcf, 'Color', 'w')

%% Komentari

% Uporediti FPR TPR i ostale metrike za razlicita obelezja
% 1-3, 2-4 visoka korelacija, redundantnost
% 2 i 9 visoka korelacija!
% Uraditi redukciju (?) dimenzija, i sopstvene vrednosti algoritam za
% "dekorelaciju" - PCA? LDA?

% Junk
% m0 = mean(ob_norm)';
% m1 = mean(neg_norm')';
% m2 = mean(pos_norm')'; 
% S1 = cov(neg_norm')';
% S2 = cov(pos_norm')';
% l = diag(abs(l));
% l = sort(l, 'descend');

% %% Analiza PCA 2
% close all
% 
% Cx = cov(ob_norm);
% [V, D] = eig(Cx);
% 
% % dV = diag(V);
% dV = abs(dV);
% 
% figure
% % bar(dV, 'b')
% subplot(221)
% bar(dV)
% hold on
% [m, ind] = maxk(dV, 3);
% plot(ind, m, 'rx', 'LineWidth', 2)
% ylabel('Apsolutna sopstvena vrednost')
% xlabel('Redni broj obelezja')
% 
% dD = diag(D);
% % dD = abs(dD);
% 
% % figure
% subplot(222)
% % bar(dD, 'b')
% bar(dD)
% hold on
% [m, ind] = maxk(dD, 3);
% plot(ind, m, 'rx', 'LineWidth', 2)
% ylabel('Apsolutna sopstvena vrednost')
% xlabel('Redni broj obelezja')
% 
% [V, D] = eigs(Cx, 16);
% 
% dV = diag(V);
% dV = abs(dV);
% 
% % bar(dV, 'b')
% subplot(223)
% bar(dV)
% hold on
% [m, ind] = maxk(dV, 3);
% plot(ind, m, 'rx', 'LineWidth', 2)
% ylabel('Apsolutna sopstvena vrednost')
% xlabel('Redni broj obelezja')
% 
% dD = diag(D);
% % dD = abs(dD);
% 
% subplot(224)
% % bar(dD, 'b')
% bar(dD)
% hold on
% [m, ind] = maxk(dD, 3);
% plot(ind, m, 'rx', 'LineWidth', 2)
% ylabel('Apsolutna sopstvena vrednost')
% xlabel('Redni broj obelezja')
% subplot(131)
% h = heatmap(abs(V(:, 1)); 'ColorScaling', 'scaledcolumns');
% % h.
% subplot(132)
% h = heatmap(abs(V(:, 2)))
% subplot(133)
% h = heatmap(abs(V(:, 3)))
% h.XLabel = 'PCA obelezje 2'
% ylabel('Izvorno obelezje')
% set(gcf, 'Color', 'w')


% cmap = [repmat([1 0 0], 5, 1)
%     repmat([1 1 1], 5, 1) % highlight all cells with values from 5-10 with white
%     repmat([0 0 0], , 1)]; % highlight all cells > 10 with black
% 
% heatmap(1:5,1:5,randi(20,[5 5]),'Colormap',cmap);
% cmap = [repmat([1, 1, 1], max(abs(V(:, 1))), 1)];
% %% Analiza PCA 1
% close all
% 
% [coeff, score] = pca(ob_norm);
% pca_ob = ob_norm * coeff(:, 1 : 3);
% 
% figure
% plot3(pca_ob(1 : N, 1), pca_ob(1 : N, 2), pca_ob(1 : N, 3), 'bo', ...
%     pca_ob(N + 1 : end, 1), pca_ob(N + 1 : end, 2), pca_ob(N + 1 : end, 3), 'rx')
% legend('Bez pukotine', 'Sa pukotinom')
% xlabel('PCA obelezje 1')
% ylabel('PCA obelezje 2')
% zlabel('PCA obelezje 3')
% % title('Prikaz obelezja u prostoru')
% axis square; box on; grid on
% set(gcf, 'Color', 'w')
% 
% pca_coeff = coeff(:, 1 : 3);
% 
% figure;
% heatmap(pca_coeff)
% % 3 8 13
%%%
% close all
% figure;
% subplot(131)
% h = heatmap(abs(V(:, 1))); colorbar off
% subplot(132)
% h = heatmap(abs(V(:, 2))); colorbar off
% subplot(133)
% h = heatmap(abs(V(:, 3))); colorbar off
