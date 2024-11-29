% CODE NUMBER ONE
disp('FIRST BLOCK OF CODE');
disp(' ');

syms Rs R1 R2 R3 R4 R5 RL 
syms Cs C1 C2 C3 C4 C5 CL
syms Ls L1 L2 L3 L4 L5 LL
syms s

Zzs = [Rs, (1/(s*Cs)), s*Ls];
Zz1 = [R1, (1/(s*C1)), R1/(1+R1*C1*s), R1+1/(C1*s), s*L1+1/(s*C1)];
Zz2 = [R2, (1/(s*C2)), R2/(1+R2*C2*s), R2+1/(C2*s), s*L2+1/(s*C2)];
Zz3 = [R3, (1/(s*C3)), s*L3, R3/(1+R3*C3*s), (s*L3+1/(s*C3)), (L3*s)/(1+L3*C3*s^2)];
Zz4 = [R4, (1/(s*C4)), s*L4, R4/(1+R4*C4*s), (s*L4+1/(s*C4)), (L4*s)/(1+L4*C4*s^2)];
Zz5 = [R5, (1/(s*C5)), s*L5, R5/(1+R5*C5*s), (s*L5+1/(s*C5)), (L5*s)/(1+L5*C5*s^2)];
ZzL = [RL, (1/(s*CL)), s*LL, RL/(1+RL*CL*s), (LL*s)/(1+LL*CL*s^2)];

m = length(Zzs);
n = length(Zz1);
o = length(Zz2);
p = length(Zz3);
q = length(Zz4);
r = length(Zz5);
st = length(ZzL);

disp(['m = ', num2str(m)]);
disp(['n = ', num2str(n)]);
disp(['o = ', num2str(o)]);
disp(['p = ', num2str(p)]);
disp(['q = ', num2str(q)]);
disp(['r = ', num2str(r)]);
disp(['st = ', num2str(st)]);
disp(' ');

%================================================================================
% CODE NUMBER TWO
disp('SECOND BLOCK OF CODE');
disp(' ');

syms Zs Z1 ZL Z5

K = [Zs, Inf, Inf, Inf, Inf, Z5, ZL];
disp(K);
disp(' ');

%================================================================================
% CODE NUMBER THREE
disp('THIRD BLOCK OF CODE');
disp(' ');

% First Column
if K(1, 1) == Inf
    c1 = Inf;
    m = 1;
else
    c1 = 1;
end

% Second Column
if K(1, 2) == Inf
    c2 = Inf;
    n = 1;
else
    c2 = 1;
end

% Third Column
if K(1, 3) == Inf
    c3 = Inf;
    o = 1;
else
    c3 = 1;
end

% Fourth Column
if K(1, 4) == Inf
    c4 = Inf;
    p = 1;
else
    c4 = 1;
end

% Fifth Column
if K(1, 5) == Inf
    c5 = Inf;
    q = 1;
else
    c5 = 1;
end

% Sixth Column
if K(1, 6) == Inf
    c6 = Inf;
    r = 1;
else
    c6 = 1;
end

% Seventh Column
if K(1, 7) == Inf
    c7 = Inf;
    st = 1;
else
    c7 = 1;
end

% Assign Matrix
Bb = sym(zeros(st*r*q*p*o*m*n, 7));
S = 0;
for i = 1:m
    for j = 1:n
        for k = 1:o
            for l = 1:p
                for ki = 1:q
                    for k2i = 1:r
                        for k3i = 1:st
                            index = k3i + S;
                            Bb(index,1) = c1*Zzs(i);
                            Bb(index,2) = c2*Zz1(j);
                            Bb(index,3) = c3*Zz2(k);
                            Bb(index,4) = c4*Zz3(l);
                            Bb(index,5) = c5*Zz4(ki);
                            Bb(index,6) = c6*Zz5(k2i);
                            Bb(index,7) = c7*ZzL(k3i); 
                        end
                        S = S + st;
                    end
                end
            end
        end
    end
end

Z = Bb;
disp(Z);
disp(' ');

%================================================================================
% CODE NUMBER FOUR 
disp('FOURTH BLOCK OF CODE');
disp(' ');

count = 0;
H = {};
K_select = [];
syms s gm;

rows_K = size(K, 1);
rows_Z = size(Z, 1);

for i = 1:rows_K
    clear Zs Z1 Z2 Z3 Z4 Z5 ZL
    clear Vop Von I1a I2a v2a v1a v1b v2b I1b I2b Vin Vip vx soln va vb Hs
    syms Vop Von I1a I2a v2a v1a v1b v2b I1b I2b Vin Vip vx va vb

    Zs = K(i, 1);
    Z1 = K(i, 2);
    Z3 = K(i, 4);
    Z2 = K(i, 3);
    Z4 = K(i, 5);
    Z5 = K(i, 6);
    ZL = K(i, 7);

    a11 = 0;
    a12 = -1/gm;
    a21 = 0;
    a22 = 0;
    b11 = a11;
    b12 = a12;
    b21 = a21;
    b22 = a22;

    %Calculate original transfer function
    eqns = [
        (Vip - va) / Zs + (Von - va) / Z3 - I1a + (vx - va) / Z1 + (Vop - va) / Z5 == 0;
        (Vin - vb) / Zs + (Vop - vb) / Z3 - I1b + (vx - vb) / Z1 + (Von - vb) / Z5 == 0;
        (va - Von) / Z3 + (vx - Von) / Z2 - I2a + (0 - Von) / ZL + (Vop - Von) / Z4 + (vb - Von) / Z5 == 0;
        (vb - Vop) / Z3 + (vx - Vop) / Z2 - I2b + (0 - Vop) / ZL + (Von - Vop) / Z4 + (va - Vop) / Z5 == 0;
        (Von - vx) / Z2 + (Vop - vx) / Z2 + (va - vx) / Z1 + (vb - vx) / Z1 + I1a + I1b + I2a + I2b == 0;
        vx + v2a == Von;
        vx + v2b == Vop;
        vx + v1a == va;
        vx + v1b == vb;
        v1a == a11 * v2a - a12 * I2a;
        I1a == a21 * v2a - a22 * I2a;
        v1b == b11 * v2b - b12 * I2b;
        I1b == b21 * v2b - b22 * I2b
    ];

    soln = solve(eqns, [Vop, Von, I1a, I2a, v2a, v1a, v1b, v2b, I1b, I2b, Vin, Vip, vx]);
    Hs = (soln.Vop - soln.Von) / (soln.Vip - soln.Vin);
    count = count + 1;
    Hs = simplify(Hs);
    
    H{count} = simplify(Hs);
    Ts = Hs;
    countn = 0;

    syms LPresult HPresult BPresult BSresult GE_APresult HPN_LPNresult

    LPresult = [];
    HPresult = [];
    BPresult = [];
    BSresult = [];
    GE_APresult = [];
    HPN_LPNresult = [];

    CountLP = 0;
    CountHP = 0;
    CountBP = 0;
    CountBS = 0;
    CountGE_AP = 0;
    CountHPN_LPN = 0;

    for kk = 1:rows_Z
        %Change the values of Hs
        Zs = Z(kk, 1);
        Z1 = Z(kk, 2);
        Z2 = Z(kk, 3);
        Z3 = Z(kk, 4);
        Z4 = Z(kk, 5);
        Z5 = Z(kk, 6);
        ZL = Z(kk, 7);

        eqns = [
        (Vip - va) / Zs + (Von - va) / Z3 - I1a + (vx - va) / Z1 + (Vop - va) / Z5 == 0;
        (Vin - vb) / Zs + (Vop - vb) / Z3 - I1b + (vx - vb) / Z1 + (Von - vb) / Z5 == 0;
        (va - Von) / Z3 + (vx - Von) / Z2 - I2a + (0 - Von) / ZL + (Vop - Von) / Z4 + (vb - Von) / Z5 == 0;
        (vb - Vop) / Z3 + (vx - Vop) / Z2 - I2b + (0 - Vop) / ZL + (Von - Vop) / Z4 + (va - Vop) / Z5 == 0;
        (Von - vx) / Z2 + (Vop - vx) / Z2 + (va - vx) / Z1 + (vb - vx) / Z1 + I1a + I1b + I2a + I2b == 0;
        vx + v2a == Von;
        vx + v2b == Vop;
        vx + v1a == va;
        vx + v1b == vb;
        v1a == a11 * v2a - a12 * I2a;
        I1a == a21 * v2a - a22 * I2a;
        v1b == b11 * v2b - b12 * I2b;
        I1b == b21 * v2b - b22 * I2b
        ];

        soln = solve(eqns, [Vop, Von, I1a, I2a, v2a, v1a, v1b, v2b, I1b, I2b, Vin, Vip, vx]);
        Hs = (soln.Vop - soln.Von) / (soln.Vip - soln.Vin);

        Hs = simplify(Hs);

        K_select{count} = [Zs, Z1, Z2, Z3, Z4, Z5, ZL];
        Z_select = K_select{count};

        temp = 0;
        [nm, dn] = numden(Hs);
        Order = polynomialDegree(dn, s);
        NumOrder = polynomialDegree(nm, s);
        TOrder = Order + NumOrder;
        
        %Assign coefficients
        %For the constant term, it does not specify the other terms as 0
        if Order == 0
            coeffs_dn = coeffs(dn, s, 'All');
            a = 0;
            b = 0;
            c = coeffs_dn;

        elseif Order == 1
            coeffs_dn = coeffs(dn, s, 'All');
            a = 0;
            b = coeffs_dn(1);
            c = coeffs_dn(2);
        
        %For 3 and more terms
        else
            coeffs_dn = coeffs(dn, s, 'All');
            a = coeffs_dn(end - 2);
            b = coeffs_dn(end - 1);
            c = coeffs_dn(end);
        end

        if a == 0 || b == 0 || c == 0 || TOrder >= 5
            temp = 1;
        end

        if temp == 1
            a = 1;
            b = 1;
            c = 1;
        end

        Q = simplify(sym((a / b) * sqrt(c / a)));
        wo_sqr = simplify(sym(c / a));
        wo = sqrt(wo_sqr);
        Bandwidth = wo / Q;
        Bandwidth = collect(Bandwidth, gm);

        %For the constant term, it does not specify the other terms as 0
        if NumOrder == 0
            coeffs_nm = coeffs(nm, s, 'All');
            bhp = 0;
            bbp = 0;
            blp = coeffs_nm;

        elseif NumOrder == 1
            coeffs_nm = coeffs(nm, s, 'All');
            bhp = 0;
            bbp = coeffs_nm(1);
            blp = coeffs_nm(2);

        %For 3 and more terms
        else
            coeffs_nm = coeffs(nm, s, 'All');
            bhp = coeffs_nm(end - 2);
            bbp = coeffs_nm(end - 1);
            blp = coeffs_nm(end);
        end

        K_HP = sym(bhp / a);
        K_HP = simplify(K_HP);
        
        K_BP = sym(bbp / a / Bandwidth);
        K_BP = simplify(K_BP);
        
        K_LP = sym(blp / a / wo^2);
        K_LP = simplify(K_LP);

        if temp == 0 && K_BP == 0 && K_LP == 0
            HP_list = {Z_select, wo, Q, K_HP};
            HPresult = [HPresult; {HP_list}];
            CountHP = CountHP + 1;
        end

        if temp == 0 && K_HP == 0 && K_LP == 0 
            BP_list = {Z_select, wo, Q, K_BP};
            BPresult = [BPresult; {BP_list}];
            CountBP = CountBP + 1;
        end

        if temp == 0 && K_HP == 0 && K_BP == 0 
            LP_list = {Z_select, wo, Q, K_LP};
            LPresult = [LPresult; {LP_list}];
            CountLP = CountLP + 1;
        end

        if temp == 0 && K_BP == 0 && K_HP ~= 0 && K_LP ~= 0
            BSresult_list = {Z_select, wo, Q, K_HP, K_LP};
            BSresult = [BSresult; {BSresult_list}];
            CountBS = CountBS + 1;
        end

        if temp == 0 && K_LP == K_HP && K_BP ~= 0 && K_LP ~= 0
            Qz = K_LP*Q/K_BP;
            Qz = simplify(Qz);
            GE_APresult_list = {Z_select, wo, Q, Qz, K_LP};
            GE_APresult = [GE_APresult; {GE_APresult_list}];
            CountGE_AP = CountGE_AP + 1;
        end

        if temp == 0 && K_LP ~= 0 && K_HP ~= 0 && K_BP == 0 && K_LP ~= K_HP
            HPN_LPNresult_list = {Z_select, wo, Q, K_HP, K_LP};
            HPN_LPNresult = [HPN_LPNresult; {HPN_LPNresult_list}];
            CountHPN_LPN = CountHPN_LPN + 1;
        end
        temp = 0;

    end
    Zs = K(i, 1);
    Z1 = K(i, 2);
    Z3 = K(i, 4);
    Z2 = K(i, 3);
    Z4 = K(i, 5);
    Z5 = K(i, 6);
    ZL = K(i, 7);

    K_select{count} = [Zs, Z1, Z2, Z3, Z4, Z5, ZL];
end
disp(count);
disp(' ');

%================================================================================
% CODE NUMBER FIVE 
disp('FIFTH BLOCK OF CODE');
disp(' ');


for fg = 1:count
    disp(K_select(fg));
    disp(['TF(', num2str(fg) ') = ', char(H{fg})]);
    
    disp(' ');
    disp(' ');
end 
disp(' ')

%================================================================================
% CODE NUMBER SIX 
disp('SIXTH BLOCK OF CODE');
disp(' ');

numbers = [CountLP, CountHP, CountBP, CountBS, CountGE_AP, CountHPN_LPN];

sizet = numbers(1);

for k = 2:length(numbers)
    sizet = gcd(sizet, numbers(k));
end

disp("Summary of all the results");
disp(' ');
disp(K_select(1));
disp(' ');
disp(['There are ', num2str(CountLP), ' Lowpass Filters']);
disp(['There are ', num2str(CountHP), ' Highpass Filters']);
disp(['There are ', num2str(CountBP), ' Bandpass Filters']);
disp(['There are ', num2str(CountBS), ' Bandstop Filters']);
disp(['There are ', num2str(CountGE_AP), ' Gain Equalizer/Allpass Filters']);
disp(['There are ', num2str(CountHPN_LPN), ' Highpass or Lowpass Notch Filters']);
disp(' ');
disp(' ');

if sizet ~= 0
    if CountLP ~= 0
        disp("LOWPASS|Zs, Z1, Z2, Z3, Z4, Z5, ZL ---|----w_o------|------Q------|------KLP-----|");
        for x = 1:CountLP
            disp(['LP_(' num2str(x) ')']);
            LP_list = LPresult{x};  

            for y = 1:length(LP_list)  
                disp(['   ', char(LP_list{y})]);  
            end
        end
        disp(' ');
    end 

    disp("-------------------------------------------------");
    disp(' ');

    if CountHP ~= 0
        disp("HIGHPASS|Zs, Z1, Z2, Z3, Z4, Z5, ZL |----w_o------|------Q------|------KHP-----|");
        for x = 1:CountHP
            disp(['HP_(' num2str(x) ')']);
            HP_list = HPresult{x};
            
            for y = 1:length(HP_list)
                disp(['   ', char(HP_list{y})]);  
            end 
        end
    end

    disp(' ');
    disp("-------------------------------------------------");

    if CountBP ~= 0 
        disp("BANDPASS|Zs, Z1, Z2, Z3, Z4, Z5, ZL |----w_o------|------Q------|------KBP-----|");
        for x = 1:CountBP
            disp(['BP_(' num2str(x) ')']);
            BP_list = BPresult{x};
            
            for y = 1:length(BP_list)
                disp(['   ', char(BP_list{y})]);  
            end 
        end
    end

    disp(' ');
    disp("-------------------------------------------------");

    if CountBS ~= 0 
        disp("NOTCH|Zs, Z1, Z2, Z3, Z4, Z5, ZL |----w_o------|------Q------|------KHP-----|------KLP----|");
        for x = 1:CountBS
            disp(['BS_(' num2str(x) ')']);
            BS_list = BSresult{x};
            
            for y = 1:length(BS_list)
                disp(['   ', char(BS_list{y})]);  
            end 
        end
    end
    
    disp(' ');
    disp("-------------------------------------------------");

    if CountGE_AP ~= 0
        disp("GE/AP|Zs, Z1, Z2, Z3, Z4, Z4, Z5, ZL |----w_o------|------Q------|------Qz-----|------K-----|");
        for x = 1:CountGE_AP
            disp(['GE_AP_(' num2str(x) ')']);
            GE_APresult_list = GE_APresult{x};
            
            for y = 1:length(GE_APresult_list)
                disp(['   ', char(GE_APresult_list{y})]);  
            end 
        end
    end
    
    disp(' ');
    disp("-------------------------------------------------");

    if CountHPN_LPN ~= 0
        disp("HPN/LPN|Zs, Z1, Z2, Z3, Z4, Z5, ZL |----w_o------|------Q------|------K_HP-----|------K_LP-----|");
        for x = 1:CountHPN_LPN
            disp(['HPN_LPN_(' num2str(x) ')']);
            HPN_LPNresult_list = HPN_LPNresult{x};
            
            for y = 1:length(HPN_LPNresult_list)
                disp(['   ', char(HPN_LPNresult_list{y})]);  
            end   
        end
    end
end
