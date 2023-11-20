addpath("./target/release");
[err, warn] = loadlibrary("libcaring");

fprintf("Hi, I am Matlab, and today we are going to add numbers\n");
me = '127.0.0.1:1234';
others = {'127.0.0.1:1235'};
%err = care_setup(me, others, length(others) - 1);
[err] = calllib("libcaring", "care_setup", me, others, length(others))
if (err ~= 0)
    fprintf("Got a nasty error:\n error code %d", err);
    quit(err);
end


%ins = [2.5, 3.5];
% res = care_sum_many(ins, length(ins));
%fprintf("Hi am Matlab, and I just did MPC, here is my result [ %f, %f ]\n", res(0), res(1));

% fprintf("Let's try some more\n");
res = 0.0;
res = care_sum(2.5);
fprintf("2.5 - 5 = %f\n", res);

res = care_sum(3.14159265359);
fprintf("pi + pi = %f\n", res);

res = care_sum(3.14159265359);
fprintf("pi - pi = %f\n", res);

res = care_sum(-1.0);
fprintf("-1 - 2 = %f\n", res);

res = care_sum(1111.1111);
fprintf("1111.1111 + 2222.2222 = %f\n", res);

res = care_sum(1111.1111);
fprintf("1111.1111 - 2222.2222 = %f\n", res);

res = care_sum(3.23e13);
fprintf("3.23e13 + 5.32e13 = %f\n", res);

res = care_sum(0.0);
fprintf("0 + 0 = %f\n", res);

res = care_sum(0.01);
fprintf("0.01 + 0.02 = %f\n", res);

calllib("libcaring", "care_takedown")

unloadlibrary("libcaring");

function [res] = care_sum(a)
    [s] = calllib("libcaring", "care_sum", a);
    if s == NaN
        quit(-1);
    end
    res = s;
end
