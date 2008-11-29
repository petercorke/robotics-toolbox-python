path = [];
N = 20;

q = (rand(N, 6)-1)*2*pi;
qd = zeros(N, 6);
qdd = zeros(N, 6);
path = [path;q qd qdd];

q = (rand(N, 6)-1)*2*pi;
qd = (rand(N, 6)-1)*2;
qdd = zeros(N, 6);
path = [path;q qd qdd];

q = (rand(N, 6)-1)*2*pi;
qd = (rand(N, 6)-1)*2;
qdd = (rand(N, 6)-1)*20;
path = [path;q qd qdd];

save path.dat path -ascii

puma560
tau = rne(p560, path)
save puma560.dat tau -ascii

puma560akb
tau = rne(p560m, path)
save puma560m.dat tau -ascii
