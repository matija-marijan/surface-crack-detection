clear
close all
clc

%% Ucitavanje obelezja i KAN pretprocesiranih podataka
close all

data = 4;

if data == 1
    positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_norm_all.csv');
    negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_norm_all.csv');
    X_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm_X_train.csv');
    X_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm_X_test.csv');
    prediction_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm_prediction_train.csv');
    prediction_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm_prediction_test.csv');
    load('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm_probs.mat')
    ind_ob = [2, 7, 16];
elseif data == 2
    positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_PCA.csv');
    negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_PCA.csv');
    X_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\PCA_X_train.csv');
    X_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\PCA_X_test.csv');
    prediction_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\PCA_prediction_train.csv');
    prediction_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\PCA_prediction_test.csv');
    load('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\PCA_probs.mat')
    ind_ob = [1, 2, 3];
elseif data == 3
    positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_LDA.csv');
    negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_LDA.csv');
    X_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\LDA_X_train.csv');
    X_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\LDA_X_test.csv');
    prediction_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\LDA_prediction_train.csv');
    prediction_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\LDA_prediction_test.csv');
    load('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\LDA_probs.mat')
    ind_ob = [1, 2, 3];
elseif data == 4
    positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_norm_all.csv');
    negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_norm_all.csv');
    X_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm4_X_train.csv');
    X_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm4_X_test.csv');
    prediction_train = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm4_prediction_train.csv');
    prediction_test = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm4_prediction_test.csv');
    load('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Data\norm4_probs.mat')
    ind_ob = [2, 4, 16];
end

N = length(negative_all);
positive_obelezja = positive_all(ind_ob, :);
negative_obelezja = negative_all(ind_ob, :);

ob1_min = min(min(negative_obelezja(1, :)), min(positive_obelezja(1, :)));
ob2_min = min(min(negative_obelezja(2, :)), min(positive_obelezja(2, :)));
ob3_min = min(min(negative_obelezja(3, :)), min(positive_obelezja(3, :)));

ob1_max = max(max(negative_obelezja(1, :)), max(positive_obelezja(1, :)));
ob2_max = max(max(negative_obelezja(2, :)), max(positive_obelezja(2, :)));
ob3_max = max(max(negative_obelezja(3, :)), max(positive_obelezja(3, :)));

figure
plot3(negative_obelezja(1, :), negative_obelezja(2, :), negative_obelezja(3, :), 'b.', ...
    positive_obelezja(1, :), positive_obelezja(2, :), positive_obelezja(3, :), 'r.')
legend('Bez pukotine', 'Sa pukotinom')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
title('Prikaz obelezja u prostoru')
axis square

%% Formiranje labela
close all

N_train = round(0.8 * N);
N_test = N - N_train;

Y_train = [zeros(1, N_train), ones(1, N_train)]';
Y_test = [zeros(1, N_test), ones(1, N_test)]';

%% Predikcija klasifikacije na trenirajucem skupu
close all

% Matrica konfuzije
fig = figure;
plotconfusion(Y_train', prediction_train')
fig.Position(3 : 4) = [300, 250];
% title('KAN predikcija na trenirajucem skupu')
title('')
set(gcf, 'Color', 'w')
ylabel('Predikcija')
xlabel('Tacna klasa')

% Iscrtavanje klasifikacione linije na trenirajucem skupu sa greskama
d = 0.15;
[x, y, z] = meshgrid(ob1_min - 0.5 : d : ob1_max + 0.5, ob2_min - 0.5 : d : ob2_max + 0.5, ob3_min - 0.5 : d : ob3_max + 0.5);
t = logical(Y_train);
% Detekcija pogresnih klasifikacija
t_pred = logical(prediction_train);
[idx_greske, ~] = find(t_pred ~= t);

figure
plot3(X_train(t, 1), X_train(t, 2), X_train(t, 3), 'bo');
hold all
plot3(X_train(~t, 1), X_train(~t, 2), X_train(~t, 3), 'rx');
plot3(X_train(idx_greske, 1), X_train(idx_greske, 2), X_train(idx_greske, 3), 'g*');
[faces, verts, ~] = isosurface(x, y, z, f, 0, x);
patch('Vertices', verts, 'Faces', faces, 'FaceColor', 'k', 'edgecolor', ...
    'k', 'FaceAlpha', 0.25, 'EdgeAlpha', 0.25);
grid on; box on; axis square
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
legend('Bez pukotine', 'Sa pukotinom', 'Pogresne klasifikacije', 'Klasifikaciona povrs')
set(gcf, 'Color', 'w')

%% Predikcija klasifikacije na testirajucem skupu
close all

% Matrica konfuzije
fig = figure;
plotconfusion(Y_test', prediction_test')
fig.Position(3 : 4) = [300, 250];
% title('KAN predikcija na testirajucem skupu')
title('')
set(gcf, 'Color', 'w')
ylabel('Predikcija')
xlabel('Tacna klasa')

% Iscrtavanje klasifikacione linije na trenirajucem skupu sa greskama
d = 0.15;
[x, y, z] = meshgrid(ob1_min - 0.5 : d : ob1_max + 0.5, ob2_min - 0.5 : d : ob2_max + 0.5, ob3_min - 0.5 : d : ob3_max + 0.5);
t = logical(Y_test);
% Detekcija pogresnih klasifikacija
t_pred = logical(prediction_test);
[idx_greske, ~] = find(t_pred ~= t);

figure
plot3(X_test(t, 1), X_test(t, 2), X_test(t, 3), 'bo');
hold all
plot3(X_test(~t, 1), X_test(~t, 2), X_test(~t, 3), 'rx');
plot3(X_test(idx_greske, 1), X_test(idx_greske, 2), X_test(idx_greske, 3), 'g*');
[faces, verts, ~] = isosurface(x, y, z, f, 0, x);
patch('Vertices', verts, 'Faces', faces, 'FaceColor', 'k', 'edgecolor', ...
    'k', 'FaceAlpha', 0.25, 'EdgeAlpha', 0.25);
grid on; box on; axis square
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
legend('Bez pukotine', 'Sa pukotinom', 'Pogresne klasifikacije', 'Klasifikaciona povrs')
set(gcf, 'Color', 'w')

















