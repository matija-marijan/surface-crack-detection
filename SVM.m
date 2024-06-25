clear
close all
clc

%% Ucitavanje prethodnih promenljivih

% save SVM_save.mat
% load SVM_save.mat

%% Ucitavanje obelezja
close all

% positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_norm_all.csv');
% negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_norm_all.csv');

% positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\Old\positive_PCA_old.csv');
% negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\Old\negative_PCA_old.csv');

% positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\Old\positive_LDA_old.csv');
% negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\Old\negative_LDA_old.csv');

positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_PCA.csv');
negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_PCA.csv');

% positive_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\positive_LDA.csv');
% negative_all = readmatrix('C:\Users\Matija\Desktop\ETF\5. GODINA\3 - MASTER RAD\Obelezja\negative_LDA.csv');

N = length(negative_all);

if size(negative_all, 1) > 3
    ind_ob = [2, 7, 16];
else
    ind_ob = [1, 2, 3];
end

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

%% Podela na skupove
close all

N_train = round(0.8 * N);
N_test = N - N_train;

p_int = randperm(N);
pos_train = positive_obelezja(:, p_int(1 : N_train));
pos_test = positive_obelezja(:, p_int(N_train + 1 : end));

n_int = randperm(N);
neg_train = negative_obelezja(:, n_int(1 : N_train));
neg_test = negative_obelezja(:, n_int(N_train + 1 : end));

%% Treniranje SVM modela (Ugradjeno)
close all

% randperm?
X_train = [pos_train, neg_train]';
Y_train = [zeros(1, N_train), ones(1, N_train)]';

% time_train = 0;
% for i = 1 : 10
% tic
%%%%% POCETAK MERENJA VREMENA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf');

%%%%% KRAJ MERENJA VREMENA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time_toc = toc;
% time_train = time_train + time_toc;
% end
% time_train = time_train / 10

sv = model.SupportVectors;
%svmclassify

%% Predikcija klasifikacije na trenirajucem skupu
close all

% Matrica konfuzije
prediction_train = predict(model, X_train);
f = figure;
plotconfusion(Y_train', prediction_train')
f.Position(3 : 4) = [300, 250];
% title('SVM predikcija na trenirajucem skupu')
title('')
set(gcf, 'Color', 'w')
ylabel('Predikcija')
xlabel('Tacna klasa')

% Iscrtavanje klasifikacione linije na trenirajucem skupu sa SV
d = 0.15;
% [x, y, z] = meshgrid(ob1_min : d : ob1_max, ob2_min : d : ob2_max, ob3_min : d : ob3_max);
[x, y, z] = meshgrid(ob1_min - 0.5 : d : ob1_max + 0.5, ob2_min - 0.5 : d : ob2_max + 0.5, ob3_min - 0.5 : d : ob3_max + 0.5);
xGrid = [x(:), y(:), z(:)];
[~ , f] = predict(model, xGrid);
f = reshape(f(:, 2), size(x));
t = logical(Y_train);

figure
plot3(X_train(t, 1), X_train(t, 2), X_train(t, 3), 'bo');
hold all
plot3(X_train(~t, 1), X_train(~t, 2), X_train(~t, 3), 'rx');
plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'g*');
[faces, verts, ~] = isosurface(x, y, z, f, 0, x);
patch('Vertices', verts, 'Faces', faces, 'FaceColor', 'k', 'edgecolor', ...
    'k', 'FaceAlpha', 0.25, 'EdgeAlpha', 0.25);
grid on; axis square; box on
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
legend('Bez pukotine', 'Sa pukotinom', 'Noseci vektori', 'Klasifikaciona povrs')
set(gcf, 'Color', 'w')

% Iscrtavanje klasifikacione linije na trenirajucem skupu sa greskama
d = 0.15;
% [x, y, z] = meshgrid(ob1_min : d : ob1_max, ob2_min : d : ob2_max, ob3_min : d : ob3_max);
[x, y, z] = meshgrid(ob1_min - 0.5 : d : ob1_max + 0.5, ob2_min - 0.5 : d : ob2_max + 0.5, ob3_min - 0.5: d : ob3_max + 0.5);
xGrid = [x(:), y(:), z(:)];
[~, f] = predict(model, xGrid);
f = reshape(f(:, 2), size(x));
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

% randperm?
X_test = [pos_test, neg_test]';
prediction = predict(model, X_test);
Y_true = [zeros(1, N_test), ones(1, N_test)]';

% Matrica konfuzije
f = figure;
plotconfusion(Y_true', prediction')
f.Position(3 : 4) = [300, 250];
% title('SVM predikcija na testirajucem skupu')
title('')
set(gcf, 'Color', 'w')
ylabel('Predikcija')
xlabel('Tacna klasa')

% Iscrtavanje klasifikacione linije na testirajucem skupu sa greskama
d = 0.15;
% [x, y, z] = meshgrid(ob1_min : d : ob1_max, ob2_min : d : ob2_max, ob3_min : d : ob3_max);
[x, y, z] = meshgrid(ob1_min - 0.5 : d : ob1_max + 0.5, ob2_min - 0.5 : d : ob2_max + 1.5, ob3_min - 0.5 : d : ob3_max + 0.5);
xGrid = [x(:), y(:), z(:)];
[~ , f] = predict(model, xGrid);
f = reshape(f(:, 2), size(x));
t = logical(Y_true);
% Detekcija pogresnih klasifikacija
t_test = logical(prediction);
[idx_greske_test, ~] = find(t_test ~= t);

figure
plot3(X_test(t, 1), X_test(t, 2), X_test(t, 3), 'bo');
hold all
plot3(X_test(~t, 1), X_test(~t, 2), X_test(~t, 3), 'rx');
plot3(X_test(idx_greske_test, 1), X_test(idx_greske_test, 2), X_test(idx_greske_test, 3), 'g*');
[faces, verts, ~] = isosurface(x, y, z, f, 0, x);
patch('Vertices', verts, 'Faces', faces, 'FaceColor', 'k', 'edgecolor', ...
    'k', 'FaceAlpha', 0.25, 'EdgeAlpha', 0.25);
grid on; axis square; box on
% title('SVM klasifikacija na testirajucem skupu')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
legend('Bez pukotine', 'Sa pukotinom', 'Pogresne klasifikacije', 'Klasifikaciona povrs')
set(gcf, 'Color', 'w')

%% JUNK - Iscrtavanje 2

% % Iscrtavanje nosecih vektora
% figure
% plot3(pos_train(1, :), pos_train(2, :), pos_train(3, :), 'r.', ...
%     neg_train(1, :), neg_train(2, :), neg_train(3, :), 'b.')
% hold on
% plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'go')
% legend('Sa pukotinom', 'Bez pukotine', 'Noseci vektori')
% title('SVM klasifikacija na trenirajucem skupu')
% set(gcf, 'Color', 'w')
% grid on
% xlabel(['Obelezje ', num2str(ind_ob(1))])
% ylabel(['Obelezje ', num2str(ind_ob(2))])
% zlabel(['Obelezje ', num2str(ind_ob(3))])
% axis square

% close all
% 
% % f = predict(model, meshgrid)
% % surf(f)?
% 
% d = 0.1;
% 
% [x, y, z] = meshgrid(min(X(:, 1)) : d : max(X(:, 1)),...
%     min(X(:, 2)) : d : max(X(:, 2)), min(X(:, 3)) : d : max(X(:, 3)));
% 
% xGrid = [x(:), y(:), z(:)];
% 
% [rez, f] = predict(model, xGrid);
% f = reshape(f(:, 2), size(x));
% rez = reshape(rez, size(x));
% % surf(f)
% 
% figure
% % plot3(X_test(t, 1), X_test(t, 2), X_test(t, 3), 'b.');
% hold on
% % plot3(X_test(~t, 1), X_test(~t, 2), X_test(~t, 3), 'r.');
% % plot3(rez)
% 
% % prikaz = [xGrid, rez];
% 
% % surf(xGrid, rez)
% vq = interp3(x, y, z, rez);
% 
% % contour3(f(:, :, 3), [0, 0, 0], 'k');
% % scatter3(x, y, f, 3)
% % mesh(x, y, f)
% % f = griddata(x, y, f)
% % surf(x, y, f)

