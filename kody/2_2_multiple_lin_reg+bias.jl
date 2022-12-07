using PyPlot
using Revise
using Distributions
using DelimitedFiles
using CSV
using DataFrames

include("pom_funkce_lin_regrese.jl")

data_folder = "../data/"

df = readdlm(data_folder*"heart.data.csv",  ',', '\n')
header = string.(df[1,:])
data = Float64.(df[2:end,:])
println(size(data))
println("Parameters : ", header)


output_base = "../vysledky/"
if !isdir(output_base)
        mkdir(output_base)
end

output_folder = output_base * "multiple_linear_regrese_bias/"
if !isdir(output_folder)
        mkdir(output_folder)
end

################################################################################

# vykresleni uvodnich statistik

export_histograms(header, data, output_folder)
plot_zavislosti(header, data, output_folder)

################################################################################

# param ... sloupce matice X pro linearni regresi
# step_w, step_b ... kroky gradientniho sestupu (GD)
# max_ter ... max pocet iteraci GD
# vypis ... vypis statistik GD po kazde xx iteraci
# resid =(loss_n - loss_(n-1))/loss_n ... zastavovaci podminka pro GD


max_iter = 1.0e8
step_b = 2.0e-8
step_w = 1.0e-6
resid = 1.0e-7
vypis = 1.0e5

X = data[:,1:end-1]
y = data[:,end]

w_init = rand(-0.01:0.01, size(X)[2])
b_init = rand(0.0:20.0)


w_final, b_final, g_norm_w, g_norm_b, loss_final, iter_final = grad_descend(X, y, w_init, b_init, step_w, step_b, max_iter, resid, vypis)


# w = vysledne vahy pro jednotlive sloupce
# vizualizace vysledku :
#        osa x ... sloupec matice X s nejvyssi vahou
#        osa y ... y (cena nemovitosti)


println(header[end]," : ")
println("Bias ... ", b_final)
for i = 1:size(X)[2]
        println("   ", header[i] ," ... w = ", w_final[i] )
end
println()


plot_lin_regrese_wBias_results(collect(1:1:size(X)[2]), X, y, w_final, b_final, header, output_folder)
