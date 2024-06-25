clear
close all
clc

%% Ucitavanje prethodnih promenljivih

% save GA_save.mat
% load GA_save.mat

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

N_ob = length(negative_all);

if size(negative_all, 1) > 3
    ind_ob = [2, 11, 16];
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

N_train = round(0.8 * N_ob);
N_test = N_ob - N_train;

p_int = randperm(N_ob);
pos_train = positive_obelezja(:, p_int(1 : N_train));
pos_test = positive_obelezja(:, p_int(N_train + 1 : end));

n_int = randperm(N_ob);
neg_train = negative_obelezja(:, n_int(1 : N_train));
neg_test = negative_obelezja(:, n_int(N_train + 1 : end));

%% Genetski algoritam - Koeficijenti polinoma

% Jedinke su koeficijenti u polinomu h(x, y):
% x, x2, x3
% y, y2, y3
% xy, x2y, xy2, x2y2
% Slobodan clan ck

% Jedan hromozom sadrzi num_coef koeficijenata, svaki sa po 8 bita 
num_coef = 11;
% Velicina populacije
N_pop = 50;
% Broj bita jednog hromozoma
L = num_coef * 8;
% Verovatnoca ukrstanja
pc = 0.9;
% Verovatnoca mutacije
pm = 0.001;
% Generacijski jaz
G = 0.8;
% Maksimalan broj generacija
max_br_gen = 100;

%% Genetski algoritam

% time_train = 0;
% for i = 1 : 10
% tic
%%%%% POCETAK MERENJA VREMENA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Inicijalna populacija
for i = 1 : N_pop
    for j = 1 : num_coef
%         dec2bin se muci sa brojevima vecim od 2^53 -> rucno
%         gen(i, 1 : L) = dec2bin(round(rand * (2 ^ L - 1)), L);
        gen(i, (j - 1) * L / num_coef + 1 : j * L / num_coef) = dec2bin(round(rand * (2 ^ (L / num_coef) - 1)), L / num_coef);
    end
end

% Obucavanje genetskog algoritma
uslov = 0;
br_gen = 0;
Fmax = [];

while uslov == 0
    br_gen = br_gen + 1;

    % Dekodovanje jedinki, skaliranje u opseg [-1, 1]
    cx = bin2dec(gen(:, 1 : L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cx2 = bin2dec(gen(:, L / num_coef + 1 : 2 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cx3 = bin2dec(gen(:, 2 * L / num_coef + 1 : 3 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;

    cy = bin2dec(gen(:, 3 * L / num_coef + 1 : 4 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cy2 = bin2dec(gen(:, 4 * L / num_coef + 1 : 5 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cy3 = bin2dec(gen(:, 5 * L / num_coef + 1 : 6 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;

    cxy = bin2dec(gen(:, 6 * L / num_coef + 1 : 7 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cx2y = bin2dec(gen(:, 7 * L / num_coef + 1 : 8 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cxy2 = bin2dec(gen(:, 8 * L / num_coef + 1 : 9 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    cx2y2 = bin2dec(gen(:, 9 * L / num_coef + 1 : 10 * L / num_coef)) / (2 ^ (L / num_coef) - 1) * 2 - 1;
    
    % Veci opseg za slobodan clan (?)
    ck = bin2dec(gen(:, 10 * L / num_coef + 1 : L)) / (2 ^ (L / num_coef) - 1) * 2 - 1;

    % Fitness funkcija - Broj tacno / pogresno klasifikovanih odbiraka
    f = zeros(N_pop, 1);
    pos_T = zeros(N_pop, 1);
    pos_N = zeros(N_pop, 1);
    neg_T = zeros(N_pop, 1);
    neg_N = zeros(N_pop, 1);
    % Predlog za fitness: Udaljenost od h(x)?

    % Celije za pamcenje pogresnih klasifikacija
    pos_N_index = cell(N_pop, 1);
    neg_N_index = cell(N_pop, 1);

    for p = 1 : N_pop
        for i = 1 : N_train
            
            x_pos = pos_train(1, i);
            y_pos = pos_train(2, i);

            h_pos = cx(p) * x_pos + cx2(p) * x_pos ^ 2 + cx3(p) * x_pos ^ 3 + ...
                cy(p) * y_pos + cy2(p) * y_pos ^ 2 + cy3(p) * y_pos ^ 3 + ...
                cxy(p) * x_pos * y_pos + cx2y(p) * x_pos ^ 2 * y_pos + cxy2(p) * x_pos * y_pos ^ 2 + cx2y2(p) * x_pos ^ 2 * y_pos ^ 2 + ...
                ck(p);

            % Ako je tacno klasifikovano, f++
            if pos_train(3, i) > h_pos
                f(p) = f(p) + 1;
                pos_T(p) = pos_T(p) + 1;
            else
                pos_N(p) = pos_N(p) + 1;
                % Pamcenje pogresnog pozitivnog odbirka
                pos_N_index{p} = [pos_N_index{p}, i];
            end

            x_neg = neg_train(1, i);
            y_neg = neg_train(2, i);

            h_neg = cx(p) * x_neg + cx2(p) * x_neg ^ 2 + cx3(p) * x_neg ^ 3 + ...
                cy(p) * y_neg + cy2(p) * y_neg ^ 2 + cy3(p) * y_neg ^ 3 + ...
                cxy(p) * x_neg * y_neg + cx2y(p) * x_neg ^ 2 * y_neg + cxy2(p) * x_neg * y_neg ^ 2 + cx2y2(p) * x_neg ^ 2 * y_neg ^ 2 + ...
                ck(p);

            if neg_train(3, i) < h_neg
                f(p) = f(p) + 1;
                neg_T(p) = neg_T(p) + 1;
            else
                neg_N(p) = neg_N(p) + 1;
                % Pamcenje pogresnog negativnog odbirka
                neg_N_index{p} = [neg_N_index{p}, i];
            end

        end
    end

    % Maksimalna vrednost fitness funkcije i pozicija tog hromozoma
    [fmax, imax] = max(f);
    Fmax = [Fmax, fmax];

    % Optimalne vrednosti koeficijenata klasifikatora
    coef = [cx(imax), cx2(imax), cx3(imax), ...
        cy(imax), cy2(imax), cy3(imax), ...
        cxy(imax), cx2y(imax), cxy2(imax), cx2y2(imax), ...
        ck(imax)];

    % Broj tacnih i netacnih klasifikacija
    posneg = [pos_T(imax), pos_N(imax), neg_T(imax), neg_N(imax)];
    
    % Formiranje nove generacije
    gen_staro = gen;
    clear gen
    N_reprodukcija = round((1 - G) * N_pop);
    N_ukrstanje = N_pop - N_reprodukcija;

    if mod(N_ukrstanje, 2) == 1
        N_ukrstanje = N_ukrstanje + 1;
        N_reprodukcija = N_reprodukcija - 1;
    end

    % Reprodukcija tockom ruleta uz elitizam
    % Direktno slanje jedinki u novu generaciju
    cc = cumsum(f);
    gen(1, 1 : L) = gen_staro(imax, 1 : L);
    for i = 2 : N_reprodukcija
        pom = rand * cc(N_pop);
        pom_i = find(sign(cc - pom) == 1, 1, "first");
        gen(i, 1 : L) = gen_staro(pom_i, 1 : L);
    end

    % Ukrstanje
    for i = 1 : N_ukrstanje / 2

        % Izbor roditelja
        pom = rand * cc(N_pop);
        pom_i = find(sign(cc - pom) == 1, 1, "first");
        roditelj1 = gen_staro(pom_i, 1 : L);

        pom = rand * cc(N_pop);
        pom_i = find(sign(cc - pom) == 1, 1, "first");
        roditelj2 = gen_staro(pom_i, 1 : L);

        % Potomci
        if rand < pc
            tu = ceil(rand * (L - 1));
            dete1 = [roditelj1(1 : tu), roditelj2(tu + 1 : L)];
            dete2 = [roditelj2(1 : tu), roditelj1(tu + 1 : L)];
        else
            dete1 = roditelj1;
            dete2 = roditelj2;
        end

        gen(N_reprodukcija + 2 * i - 1, 1 : L) = dete1;
        gen(N_reprodukcija + 2 * i, 1 : L) = dete2;

    end

    % Mutacija
    % Nasumicno biranje pm * N * L bitova u celoj generaciji koji ce mutirati
    % (umesto trazenje nekog bita sa pm verovatnocom u svakoj jedinki)
    for i = 1 : round(pm * N_pop * L)
        N_i = ceil(rand * N_pop);
        L_i = ceil(rand * L);
        gen(N_i, L_i) = num2str(1 - str2num(gen(N_i, L_i)));
    end

    % Prestanak rada algoritma:
    % Dostignut maksimalan broj generacija, ili
    % Najboljih 10 jedinki je previse slicno
    fs = sort(f);
    if br_gen > max_br_gen || abs(fs(N_pop) - fs(N_pop - 10)) < 0.001
        uslov = 1;
    end
end

%%%%% KRAJ MERENJA VREMENA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time_toc = toc;
% time_train = time_train + time_toc;
% end
% time_train = time_train / 10

%% Iscrtavanje najboljih vrednosti fitness funkcije kroz generacije
close all

procenat = Fmax / N_train / 2 * 100;
figure
plot(procenat, 'LineWidth', 1)
% title('Najbolje vrednosti fitness funkcije kroz generacije')
xlabel('Generacija')
ylabel('Fitness funkcija [%]')
ylim([90 100])
xlim([0 br_gen + 1])
grid on
set(gcf, 'Color', 'w')
disp(['Fmax je ', num2str(Fmax(end) / N_train / 2 * 100), '%.'])

%% Predikcija klasifikacije na trenirajucem skupu
close all

prediction = [zeros(1, posneg(1)), ones(1, posneg(2)), ones(1, posneg(3)), zeros(1, posneg(4))];
Y_true = [zeros(1, N_train), ones(1, N_train)];

f = figure;
plotconfusion(Y_true, prediction)
f.Position(3 : 4) = [300, 250];
% title('GA predikcija na trenirajucem skupu')
title('')
ylabel('Predikcija')
xlabel('Tacna klasa')
set(gcf, 'Color', 'w')

% Iscrtavanje klasifikacione linije na trenirajucem skupu
pos_N_imax = pos_N_index{imax};
neg_N_imax = neg_N_index{imax};

figure
plot3(neg_train(1, :), neg_train(2, :), neg_train(3, :), 'bo', ...
    pos_train(1, :), pos_train(2, :), pos_train(3, :), 'rx')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
axis square; box on, grid on
hold all

x = ob1_min - 0.5 : 0.075 : ob1_max + 0.5;
y = ob2_min - 0.5 : 0.075 : ob2_max + 0.5;
[X, Y] = meshgrid(x, y);

h = coef(1) * X + coef(2) * X .^ 2 + coef(3) * X .^ 3 + ...
    coef(4) * Y + coef(5) * Y .^ 2 + coef(6) * Y .^ 3 + ...
    coef(7) * X .* Y + coef(8) * X .^ 2 .* Y + coef(9) * X .* Y .^ 2 + coef(10) * X .^ 2 .* Y .^ 2 + ...
    coef(11);

scatter3(pos_train(1, pos_N_imax), pos_train(2, pos_N_imax), pos_train(3, pos_N_imax), 'g*')
scatter3(neg_train(1, neg_N_imax), neg_train(2, neg_N_imax), neg_train(3, neg_N_imax), 'g*')
mesh(X, Y, h, 'FaceColor', 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'k', 'EdgeAlpha', 0.75)
zlim([ob3_min - 0.5, ob3_max + 1])

% title('GA klasifikacija na trenirajucem skupu')
legend('Bez pukotine', 'Sa pukotinom', 'Pogresne klasifikacije', '', 'Klasifikaciona povrs')
set(gcf, 'Color', 'w')

%% Predikcija klasifikacije na testirajucem skupu
close all

pos_T_test = 0;
pos_N_test = 0;
neg_T_test = 0;
neg_N_test = 0;
pos_N_test_index = [];
neg_N_test_index = [];

for i = 1 : N_test

    x_pos = pos_test(1, i);
    y_pos = pos_test(2, i);

    h_pos = coef(1) * x_pos + coef(2) * x_pos ^ 2 + coef(3) * x_pos ^ 3 + ...
        coef(4) * y_pos + coef(5) * y_pos ^ 2 + coef(6) * y_pos ^ 3 + ...
        coef(7) * x_pos * y_pos + coef(8) * x_pos ^ 2 * y_pos + coef(9) * x_pos * y_pos ^ 2 + coef(10) * x_pos ^ 2 * y_pos ^ 2 + ...
        coef(11);

    if pos_test(3, i) > h_pos
        pos_T_test = pos_T_test + 1;
    else
        pos_N_test = pos_N_test + 1;
        pos_N_test_index = [pos_N_test_index, i];
    end

    x_neg = neg_test(1, i);
    y_neg = neg_test(2, i);

    h_neg = coef(1) * x_neg + coef(2) * x_neg ^ 2 + coef(3) * x_neg ^ 3 + ...
        coef(4) * y_neg + coef(5) * y_neg ^ 2 + coef(6) * y_neg ^ 3 + ...
        coef(7) * x_neg * y_neg + coef(8) * x_neg ^ 2 * y_neg + coef(9) * x_neg * y_neg ^ 2 + coef(10) * x_neg ^ 2 * y_neg ^ 2 + ...
        coef(11);

    if neg_test(3, i) < h_neg
        neg_T_test = neg_T_test + 1;
    else
        neg_N_test = neg_N_test + 1;
        neg_N_test_index = [neg_N_test_index, i];
    end
end

prediction = [zeros(1, pos_T_test), ones(1, pos_N_test), ones(1, neg_T_test), zeros(1, neg_N_test)];
Y_true = [zeros(1, N_test), ones(1, N_test)];

f = figure;
plotconfusion(Y_true, prediction)
f.Position(3 : 4) = [300, 250];
% title('GA predikcija na testirajucem skupu')
title('')
ylabel('Predikcija')
xlabel('Tacna klasa')
set(gcf, 'Color', 'w')

% Iscrtavanje klasifikacione linije na testirajucem skupu
figure
plot3(neg_test(1, :), neg_test(2, :), neg_test(3, :), 'bo', ...
    pos_test(1, :), pos_test(2, :), pos_test(3, :), 'rx')
xlabel(['Obelezje ', num2str(ind_ob(1))])
ylabel(['Obelezje ', num2str(ind_ob(2))])
zlabel(['Obelezje ', num2str(ind_ob(3))])
axis square; box on; grid on
hold all

x = ob1_min - 0.5 : 0.075 : ob1_max + 0.5;
y = ob2_min - 0.5 : 0.075 : ob2_max + 0.5;
[X, Y] = meshgrid(x, y);

h = coef(1) * X + coef(2) * X .^ 2 + coef(3) * X .^ 3 + ...
    coef(4) * Y + coef(5) * Y .^ 2 + coef(6) * Y .^ 3 + ...
    coef(7) * X .* Y + coef(8) * X .^ 2 .* Y + coef(9) * X .* Y .^ 2 + coef(10) * X .^ 2 .* Y .^ 2 + ...
    coef(11);

scatter3(pos_test(1, pos_N_test_index), pos_test(2, pos_N_test_index), pos_test(3, pos_N_test_index), 'g*')
scatter3(neg_test(1, neg_N_test_index), neg_test(2, neg_N_test_index), neg_test(3, neg_N_test_index), 'g*')
mesh(X, Y, h, 'FaceColor', 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'k', 'EdgeAlpha', 0.75)
zlim([ob3_min - 0.5, ob3_max + 1])

% title('GA klasifikacija na testirajucem skupu')
legend('Bez pukotine', 'Sa pukotinom', 'Pogresne klasifikacije', '', 'Klasifikaciona povrs')
set(gcf, 'Color', 'w')

%% Junk

% pos_train = positive_obelezja(:, 1 : round(0.7 * N_ob));
% pos_test = positive_obelezja(:, round(0.7 * N_ob) + 1 : end);
% neg_train = negative_obelezja(:, 1 : round(0.7 * N_ob));
% neg_test = negative_obelezja(:, round(0.7 * N_ob) + 1 : end);
% 
% N_train = length(pos_train);
% N_test = length(pos_test);
