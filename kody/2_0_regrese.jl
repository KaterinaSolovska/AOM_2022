using PyPlot
using Revise
using Distributions
using DelimitedFiles
using CSV
using DataFrames

include("pom_funkce.jl")


output_base = "../vysledky/"
if !isdir(output_base)
        mkdir(output_base)
end

################################################################################

# n = celkovy pocet bodu, [a,b] = rozsah hodnot x, y = sin(x) + e, e ¬ N(0,s), s = rozptyl sumu

n = 20
s = 0.075

a = -3.0
b = 3.0

x = rand(a:0.01:b, n)
y = sin.(x)

e = rand(Normal(0, s), n)     #  e ~ N(0, s)
y_e = y+e

output_folder = output_base * "regrese_n="*string(n)*"_s="*string(s)*"_a="*string(a)*"_b="*string(b)*"/"
if !isdir(output_folder)
        mkdir(output_folder)
end

################################################################################
# d = stupen polynomu
# eps = krok grad. sestupu (GD), resid =(loss_n - loss_(n-1))/loss_n ... zastavovaci podminka pro GD
# iter_max = max pocet iteraci GD, iter_vypis = vypis statistik GD, l = vaha regularizacniho clenu

d = 5

eps = 1.e-7
resid = 1.0e-8
iter_max = 1.0e7
iter_vypis = 5.0e5

l_set = [0, 1.0e-1]

w_init = rand(-0.001:0.001, d+1)
y_init = P(w_init, x)

for l in l_set
        println("Setting: degree = "*string(d)*", regul. = "*string(l))
        w_final, grad_final, loss_final, i_final = grad_descend(x, y_e, w_init, d+1, eps, resid, iter_max, iter_vypis, l)
        println("w final = ", w_final)
        println()

        plot_init_regrese(a, b, x, y_e, d, output_folder)
        plot_results_regrese(a, b, x, y_e, w_init, w_final, grad_final, i_final, loss_final, l, d, output_folder)
end
